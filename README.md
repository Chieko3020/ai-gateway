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

### 请求能力的边界

- **流式（`stream: true`）真透传 + 语义缓存**：上游 SSE 逐块转发给客户端（`llm_client` 用
  write 回调，不再整段缓冲），`Content-Type` 按上游原样透传（`text/event-stream`），长度语义用
  `Transfer-Encoding: chunked`。
  - **流式请求同样走语义缓存**：命中时回放**当初记下的上游原始 SSE 字节**（而不是把缓存的
    文本重新合成事件——流式响应没有可以改写的 JSON 容器，合成的流会自己造
    `finish_reason`/`usage`/事件切分，与上游字节不等价）。命中响应带 `X-Cache: hit`，流首附一条
    `: cache hit` 注释行（SSE 规范允许、客户端默认忽略）。未命中时边透传边捕获原始字节，
    **只有正常流完（含 `[DONE]`）才回填**——客户端中途断开留下的半截答案不会写进缓存。
    命中的条目若没有 SSE 字节（非流式路径写入的条目），按未命中回源。
  - 输出过滤按 **SSE 事件边界**逐条判定：命中拦截规则的那一条事件被丢弃，其余事件照常透传
    （不会把整段流替换成错误 JSON）。启用 `filter.max_output_chars` 时，到达上限会补发一个
    `data: [DONE]` 并停止透传，让客户端拿到明确的结束标志。客户端断开或写死线到点时，
    网关立即中停上游传输。
  ```bash
  curl -N http://127.0.0.1:9000/v1/chat/completions \
    -H 'Content-Type: application/json' \
    -d '{"model":"deepseek-flash","stream":true,"messages":[{"role":"user","content":"你好"}]}'
  ```
  使用 `stream: true` 的客户端（例如 DSH 的 `pi-ai` 层在 openai-completions 里硬编码 `stream: true`）
  可以直接把本网关当作 provider。
- **HTTP keep-alive**：HTTP/1.1 默认复用连接（`Connection: close` 的请求仍会被关闭；
  HTTP/1.0 默认不复用）。空闲连接由 `server.idle_timeout_seconds`（默认 30s）回收。
  流式响应用 chunked 承载，因此**流式响应结束后连接同样可复用**。
- **工具调用类请求（`tools` / `functions` / `tool` 角色 / `tool_calls` / `function_call`）走旁路**：不查缓存、
  不写缓存、不参与请求合并，但**仍执行输入/输出安全过滤**，并按上游状态码原样透传。
- **落盘向量带 embedding 指纹**：`cache/lru_store.json` 里记录产生向量的模型与词表哈希、维度、
  分词器标识。换了模型或改了分词之后加载，指纹不一致的**向量会被丢弃、文本条目保留**
  （日志 WARN 提示），并按新模型重建索引——避免"旧向量当新向量用"造成的静默错误命中。
  旧格式（无指纹）文件同样能读入：此时向量按不可考处理（丢弃）并打印 WARN。

### 技术特性
- **并发模型**: 单 Reactor + 线程池，主线程管理连接，线程池处理缓存和 LLM 转发
- **向量检索**: 按论文实现 HNSW 图索引（分层结构、几何分布层数、启发式邻居选择、邻居满时收缩重选）。实测（见 `tests/recall_bench`）：320 向量下自检索 top-1 **100%**（320/320）、top-3 召回率 99.7%；1000 / 5000 / 10000 向量下自检索 top-1 均为 **100%**，top-3 召回率 99.7% / 93.8% / 81.7%（`ef_search=50` 固定时规模增大导致束搜索覆盖不足，属 HNSW 固有特性；缓存只需 top-1，故不影响命中），相对暴力检索加速比 1.16x / 2.74x / 4.75x
- **存储引擎**: LRU + TTL 缓存管理，JSON 持久化
- **嵌入推理**: C++ ONNX Runtime 进程内 INT8 量化推理，零外部依赖
- **模型量化**: `scripts/quantize.py` — HuggingFace → FP32 ONNX → 动态量化 INT8 (~90MB→~23MB)
- **连接池**: libcurl Keep-Alive 复用 TCP 连接（每线程一个 `CURL*`，互不共享）

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
├── include/
│   ├── common/                  # 公共组件
│   │   ├── config.h             #   JSON 配置加载
│   │   ├── logger.h             #   日志宏
│   │   ├── log_file.h           #   日志输出与轮转
│   │   ├── types.h              #   公共类型（ErrorCode）
│   │   ├── curl_client.h        #   RAII libcurl 封装
│   │   ├── singleflight.h       #   同义在途请求合并
│   │   └── thread_pool.h        #   线程池
│   ├── server/                  #   模块1: HTTP 接入
│   │   ├── http_server.h        #     epoll ET + 线程池调度 + ResponseWriter
│   │   ├── connection_handler.h #     连接生命周期与 keep-alive 决策
│   │   ├── request.h            #     HTTP/1.1 请求解析
│   │   ├── response.h           #     HTTP 响应构造
│   │   ├── filter.h             #     安全过滤器（含 SSE 按事件过滤）
│   │   ├── router.h             #     URL 路由
│   │   ├── metrics.h            #     /metrics 的 Prometheus 文本渲染
│   │   ├── sse_usage.h          #     旁路解析 SSE 里的 token 用量
│   │   └── sse_capture.h        #     旁路解析 SSE，还原回答文本（流式回填用）
│   ├── cache/                   #   模块2: 语义缓存
│   │   ├── cache_engine.h       #     缓存协调器（命名空间隔离 + 实体否决）
│   │   ├── lru_store.h          #     LRU + TTL 存储 + JSON 持久化（含原始 SSE 字节）
│   │   ├── hnsw_index.h         #     HNSW 图索引（header-only）
│   │   ├── onnx_embedding.h     #     ONNX Runtime 进程内推理 + WordPiece 分词
│   │   ├── entity_tokens.h      #     实体标记提取与不对称差集判定
│   │   └── embedding_fingerprint.h  # 向量指纹（模型/词表/分词器标识）
│   ├── gateway/                 #   模块3: 请求管道
│   │   └── pipeline.h           #     输入过滤 → 流式/缓存/合并 → 上游 → 统计 → 输出过滤
│   ├── backend/                 #   模块4: LLM 后端
│   │   └── llm_client.h         #     LLM API 客户端（含 SSE 流式回调）
│   └── stats/                   #   模块5: 统计
│       └── stats.h              #     命中率/token/费用/延迟统计
├── src/                         # 源文件（与 include/ 同构）
│   ├── main.cpp                 #   入口：组装 + 启动 + 信号处理 + 优雅停机
│   ├── common/  server/  cache/  gateway/  backend/  stats/
├── model/                       # 内嵌模型文件
│   ├── model_int8.onnx          #   INT8 量化 bge-small-zh (23MB)
│   └── vocab.txt                #   词表 (107KB)
├── tests/                       # 单元测试（23 个 ctest 目标）
├── config/
│   └── gateway.example.json     #   配置模板
├── scripts/
│   ├── ai-gateway.service       #   systemd 服务
│   ├── benchmark.py             #   综合压测脚本
│   ├── integration_pipeline_test.sh  # 真二进制端到端集成验证（A–F 六段）
│   ├── gw_probe.py              #   集成验证用的探针 / mock 上游
│   ├── eval_semantic_cache.py   #   公开数据集上的缓存质量评测
│   ├── fetch_eval_datasets.sh   #   下载 LCQMC / PAWS-X
│   └── quantize.py              #   FP32 → INT8 动态量化
├── .gitignore
├── CMakeLists.txt
└── README.md
```

### 核心模块

#### server — HTTP 接入层
- **HttpServer**: epoll ET 主循环 + 线程池调度
- **ConnectionHandler**: 连接生命周期（keep-alive 决策、借出与归还）
- **Filter**: 安全过滤（反注入 / URL / 关键词 / 长度），流式输出按 **SSE 事件边界**逐条判定
- **Router**: URL 路由表（METHOD + path 精确匹配）
- **Request/Response**: HTTP/1.1 解析与响应构造

#### gateway — 请求管道
- **Pipeline**: 一次请求的完整生产路径（输入过滤 → 工具请求旁路 → 流式分支 → 语义缓存 →
  请求合并 → 上游转发 → 写缓存 → 统计 → 输出过滤）。抽成独立库是为了让单测链接**同一份**
  对象代码，而不是测试里另写一份逻辑。

#### cache — 语义缓存引擎
- **CacheEngine**: 编排缓存流程（向量化、Top-K 检索、阈值 + 命名空间 + 实体一致性判定）
- **HnswIndex**: HNSW 图索引（header-only；按论文实现几何分布层数、启发式邻居选择与邻居饱和收缩，
  `shared_mutex` 保护读写并发，visited 标记线程本地复用）
- **LruStore**: LRU + TTL 内存缓存 + JSON 持久化（锁内只取快照，序列化与落盘在锁外；
  临时文件 + `fsync` + `rename` 原子替换）。条目同时保存回答文本与**上游原始 SSE 字节**
- **OnnxEmbedding**: C++ ONNX Runtime 进程内 INT8 推理 + 对齐官方的 BERT WordPiece 分词
- **EntityTokens / EmbeddingFingerprint**: 实体标记不对称差集否决；向量指纹（换模型 / 改分词后
  旧向量不再被当作新向量使用）

#### backend — LLM 转发
- **LlmClient**: LLM API 客户端（Keep-Alive 连接池复用 TCP；流式路径用 write 回调逐块交付，
  并支持按"两次数据的间隔"判定的上游空闲死线）

#### stats — 统计
- **Stats**: 命中率 / token / 费用 / 延迟统计，每 60s 定期输出

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
│    3. max(cosine) >= 0.85                                        │
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
| `embedding.model_path` | model/model_int8.onnx | 进程内 ONNX 模型路径 |
| `embedding.vocab_path` | model/vocab.txt | WordPiece 词表路径 |
| `embedding.dim` | 512 | 向量维度；与模型实际输出不符时启动即失败 |
| `cache.enabled` | true | 启用语义缓存 |
| `cache.similarity_threshold` | 0.85 | 余弦相似度阈值 |
| `cache.entity_veto` | true | 实体一致性否决：候选与查询的数字 / 大写缩略语 / 混合标识符存在不对称差集时否决该次命中（关掉可退回纯阈值判定，用于 A/B 对照） |
| `cache.max_entries` | 10000 | 最大缓存条目 |
| `cache.ttl_days` | 7 | 缓存过期天数 |
| `filter.max_input_chars` | 500 | 输入最大字符 |
| `filter.max_output_chars` | 0 | 输出最大字符（0 = 不截断）。流式响应按**事件**累计，到达上限后补发一个 `data: [DONE]` 并停止透传——截断也必须给客户端一个明确的流结束标志 |
| `filter.block_urls` | true | 拦截 URL |
| `server.max_connections` | 256 | 并发连接上限（超出回 503） |
| `server.idle_timeout_seconds` | 30 | 连接空闲超时（秒） |
| `server.stream_idle_timeout_seconds` | 60 | 流式请求的**上游空闲死线**：上游连续多久不发数据就中停它（按"两次数据的间隔"判定，不是平均速率） |
| `log.sample_every` | 1 | 每请求 INFO 采样率（1 = 全量，N = 每 N 条留 1 条） |
| `log.max_bytes` | 10485760 | 单文件日志上限（字节），达到后切分；0 = 关闭轮转 |
| `log.keep_files` | 5 | 日志历史保留份数（不含当前文件） |
| `cost.input_per_1k` | 0.001 | 输入单价（元/1K tokens），用于费用估算 |
| `cost.output_per_1k` | 0.001 | 输出单价（元/1K tokens）；缺省与旧口径一致（统一 0.001） |

API Key 通过 `config/gateway.env`（systemd `EnvironmentFile`）或环境变量 `LLM_API_KEY` 注入。

### 可观测性

**`GET /metrics`** — Prometheus 文本格式（`text/plain; version=0.0.4`），与
`POST /v1/chat/completions` 是两条独立路由，新增该端点不改变既有 POST 行为：

```bash
curl -s http://127.0.0.1:3003/metrics
```

```yaml
# prometheus.yml
scrape_configs:
  - job_name: ai-gateway
    static_configs:
      - targets: ["127.0.0.1:3003"]
```

暴露 `ai_gateway_requests_total`（可缓存流量，不含旁路与合并）、`ai_gateway_cache_hits_total`
/ `_misses_total`、`ai_gateway_cache_hit_ratio`、`ai_gateway_cache_bypassed_total`、
`ai_gateway_cache_merged_total`、`ai_gateway_cache_writes_total`（回填次数）、
`ai_gateway_tokens_{prompt,completion,saved}_total`、
`ai_gateway_cost_yuan_total` / `ai_gateway_cost_saved_yuan_total`、延迟分位数
`ai_gateway_latency_milliseconds{quantile="0.5|0.95|0.99"}`、旁路延迟与线程池队列深度。

流式流量单列一组：`ai_gateway_streams_total`（含正常完成与中止）、`ai_gateway_streams_aborted_total`
（上游空闲超时 / 客户端断开 / 写死线）、`ai_gateway_streams_without_usage_total`（上游没给 usage）、
`ai_gateway_stream_cache_hits_total`（命中并回放缓存的流）。

口径与日志里的 `[STATS]` 摘要一致，且只含聚合数值，不含任何请求内容或键哈希。
注意"可缓存流量"的口径：经过缓存查询的流式请求计入命中率分母（命中记 `hits`，回源记 `misses`），
工具调用类请求仍记 `bypassed`——把两者混起来会让命中率的分子分母口径不一致。

**日志轮转** — `gateway.log` 达到 `log.max_bytes` 后改名为 `gateway.log.1`，旧的依次
后移，最老一份删除，共保留 `log.keep_files` 份；进程启动时若发现当前文件已超限会先归档。
设 `log.max_bytes: 0` 可退回"单文件一直追加"。轮转出的文件与当前文件同为 `0600`。

## 测试

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
# 23/23 目标（Release 与 Debug 各跑一遍都是全绿）

# 集成验证：真二进制 + 真 TCP + mock 上游，A–F 六段共 30 条断言
bash scripts/integration_pipeline_test.sh ./build/src/ai-gateway

# HNSW 召回率基准（手动运行，不纳入 ctest）
cmake --build build --target recall_bench
# 真实数据集模式（ONNX 编码后建索引）
./build/tests/recall_bench scripts/datasets/synthetic.jsonl model/model_int8.onnx model/vocab.txt
# 合成向量模式（规模曲线：抽样 200 查询与暴力检索对比）
./build/tests/recall_bench --synthetic 10000 --dim 512 --queries 200

# 语义缓存质量评测（公开数据集，先下载）
bash scripts/fetch_eval_datasets.sh
python3 scripts/eval_semantic_cache.py --limit 3000
```

两点判据上的说明：

- `test_pipeline` 链接的是与可执行文件**同一份**请求管道对象代码（`ai_gateway_pipeline`），
  覆盖的是生产路径本身，而不是测试里另写的一份逻辑。此前管道只存在于 `main.cpp`，
  于是 `drain`、请求合并的命名空间隔离、keep-alive 决策三处缺陷都落在零覆盖的代码层。
- `integration_pipeline_test.sh` 每一段都先做**监听者校验**（端口 inode ↔ 进程 fd 比对），
  避免端口被残留进程占着、测试却连到别的实例上得出假结论。

## 实测性能

> 测试环境：2 vCPU / 2GB 内存的 VPS。压测端与被测服务**同机环回**，服务 `taskset -c 0`、压测端 `taskset -c 1`。
> 上游：DeepSeek v4-flash（公网 API）；Embedding：ONNX bge-small-zh-v1.5 INT8 (512d)，进程内推理。
> 这些绝对值都带环境依赖（环回压测不含真实网络），引用时请连同环境一起引用。

### 语义匹配质量（公开数据集）

缓存判定的质量用公开数据集量，而不是自造的相似句集合——自造集的重复度会把命中率抬到不可信的高位。

| 判定策略（LCQMC 3000 对） | 召回 | 误命中 | F1 |
|---|---|---|---|
| 0.85 + 实体一致性否决（当前默认） | 87.8% | **29.6%** | 80.4% |
| 0.80 纯阈值 | 94.4% | 49.2% | 77.0% |

LCQMC 的"相似"包含问答对关系，当作"同义改写"用会天然偏高，因此**误命中率**才是关键指标：
0.80 时近一半的"命中"是错的；0.85 叠加实体一致性否决（数字 / 大写缩略语 / 混合标识符存在
不对称差集即否决）之后降到 29.6%。PAWS-X 上精确率在所有阈值下都只有 43–45%，说明该量级的
embedding 对"语序颠倒 / 主宾互换"这类深层改写无能为力——这也是加上实体否决的原因。
阈值扫描与逐条明细由 `scripts/eval_semantic_cache.py` 输出。

### 延迟

| 指标 | 合成集（320 条 × 2 轮） | 真实集（92 条 × 2 轮） |
|---|---|---|
| 命中 p50 / p95 | 13ms / 17ms | 22–25ms / 51–53ms |
| 未命中 p50 / p95 | 808ms / 1074ms | 731ms / 1065ms |
| 命中率 R1 / R2 / 合计 | 25.6% / 100% / 62.8% | 15.4% / 100% / 57.7% |

口径说明（这组数字来自 `results/replay_*_fixed.json`）：阈值 0.80、**纯阈值**、未开实体否决，
两轮均为"先写入再回放"。R1 是首次提问，同义改写还要跨过阈值才算命中，所以它反映的是**语义
匹配的严格程度**，不是缓存效率；R2 是重复提问，100% 说明缓存写入与检索链路本身是通的。
未命中延迟含公网 LLM API 往返，因此它衡量的是上游而不是网关。

真实上游口径下的回放（824 次请求，阈值 0.85 + 实体否决）：命中 469 / 未命中 355
（**56.9%**），省下 **30,287 tokens**。这里 `saved` 是按已发生未命中的平均调用量外推的
**估算值**，对外引用用 token 数比用金额稳（金额随上游定价变动）。

### 流式请求

| 指标 | 首次（回源 + 回填缓存） | 第二次（命中并回放原始 SSE 字节） |
|---|---|---|
| 端到端 | 1.444s | **0.014s** |
| 客户端收到的字节 | 34,963（含 `[DONE]`） | 34,976（= 首次 + 注释行，去掉注释行后**逐字节相同**） |

同一条链路用 OpenAI 兼容 SDK 侧（`@earendil-works/pi-ai`）复测：TTFT **2410ms → 19ms**，
两次的事件序列完全一致（`thinking_*` → `text_start` → `text_delta` × 81 → `text_end` → `done`），
usage 完整。

### 内存

| 状态 | RSS | 说明 |
|---|---|---|
| 空载 | 65.7 MB | 含 23MB INT8 模型 |
| 灌满 1 万条缓存 | 138.1 MB | 落盘 JSON 28.1 MB |
| 保存窗口 | 158.6 MB | 序列化快照的瞬时占用，不回落 |
| 灌入过程峰值 | 264–320 MB | 与并发灌入的分配峰值有关；已确认无泄漏 |

### HNSW 索引

自检索（把库里的向量当查询）top-1：320 向量 **100%**（320/320），1k / 5k / 10k 均 **100%**；
top-3 召回率 99.7% / 99.7% / 93.8% / 81.7%，相对暴力检索的加速比 1.16x / 2.74x / 4.75x。
`ef_search=50` 固定时规模增大会让束搜索覆盖不足（HNSW 固有特性），而缓存只需要 top-1。

这一项与早期版本的数字差别很大：修复前 320 向量自检索 top-1 只有 30.9%，根因是"邻居饱和时
直接放弃反向连接"，导致大量节点没有入边、在图中不可达。补上论文的
`SELECT-NEIGHBORS-HEURISTIC`（多样性筛选）与"满时收缩重选"后恢复到 100%。
