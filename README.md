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
- **安全过滤**: 请求/响应双向过滤（反注入检测、URL 拦截、关键词屏蔽、长度截断）。
  **定位说明**：这些规则都是模式匹配，作用是**减少无意的暴露面**（模型顺手贴出链接、
  出现明确不想转发的词），**不构成安全边界**——对有意规避（变形 URL、拆字、编码、
  把信息藏进看似正常的文本）天然无效，而且这一点与命中缓存无关（两条路径走同一套规则）。
  真要防外泄/诱导，需要结构性手段（出站 URL 一律改写为白名单跳转页），而不是继续加规则
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
    （不会把整段流替换成错误 JSON）。对**可能构成 URL 的内容**还会启用**跨事件判定**——
    违规串被 delta 边界劈开时（前半在一个事件、后半在下一个）也能拦下。被拦时发一个
    `data: {"error":"response filtered"}` 事件再加 `[DONE]`，让客户端正常收尾，而不是看到
    "连接异常中断"。启用 `filter.max_output_chars` 时，到达上限会补发一个 `data: [DONE]`
    并停止透传。客户端断开或写死线到点时，网关立即中停上游传输。
  - **命中缓存的流式响应同样要过输出过滤**，判定口径与写入侧**完全一致**：缓存里存的是
    上游**原文**（不是过滤后的结果），不重过滤的话，"先让含 URL 的答案入缓存、再命中"就能
    绕过输出过滤；被拦时在**写响应头之前**返回 502，客户端看到的是干净的 502。
  - 边界如实说明：流式**不可撤回**——已经写出去的字节收不回来，所以流式过滤只能"拦截后续"，
    达不到非流式"整段拦下、客户端什么都看不到"的强度。
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
  **另一种旧格式的后果要特别注意**：若条目连 `src`（来源文本）字段都没有（早期版本只写 `ctime`/`key`/`value`），
  `LruStore` 在重算向量时会直接跳过它们（`for_each_missing_embedding` 遇空 `source` 即 `continue`），
  于是**重建后的索引可能是 0 向量**，而 `index_ready` 仍会被置 1 —— 外部表现为"语义缓存静默失效、
  只剩精确命中可走"。线上确实出现过该状态（journal 里 `cache: index rebuilt, 0 vectors (0 built + 0 backfilled)`），
  排查时先看这一行的计数与 `/metrics` 的 `index_vectors`；这类旧条目无法重回语义索引，只能清空后从零积累。

### 技术特性
- **并发模型**: 单 Reactor + 线程池，主线程管理连接，线程池处理缓存和 LLM 转发
- **向量检索**: 按论文实现 HNSW 图索引（分层结构、几何分布层数、启发式邻居选择、邻居满时收缩重选）。实测（见 `tests/recall_bench`）：320 向量下自检索 top-1 **100%**（320/320）、top-3 召回率 99.7%；1000 / 5000 / 10000 向量下自检索 top-1 均为 **100%**，top-3 召回率 99.7% / 93.8% / 81.7%（`ef_search=50` 固定时规模增大导致束搜索覆盖不足，属 HNSW 固有特性；缓存只需 top-1，故不影响命中），相对暴力检索加速比 1.17x / 2.73x / 4.70x（完整四行见下文「实测性能」表，含真实集 320 的 1.31x）
- **存储引擎**: LRU + TTL 缓存管理，JSON 持久化
- **嵌入推理**: C++ ONNX Runtime 进程内 INT8 量化推理，零外部依赖
- **模型量化**: `scripts/quantize.py` — HuggingFace → FP32 ONNX → 动态量化 INT8 (~90MB→~23MB)
- **连接池**: libcurl Keep-Alive 复用 TCP 连接（每线程一个 `CURL*`，互不共享）
- **索引生命周期**: 向量索引在后台线程构建（1 万条真实 embedding 约 7s），**构建期间进程即可服务**，
  语义检索暂时退化为线性扫描（万条约 15ms，仍能命中），完成后原子交换并自动恢复；
  运行中的重建（幽灵率或过期清理触发）同样不阻塞检索，因为它用旧索引继续服务
- **崩溃安全的持久化**: 锁内取快照 + 锁外序列化 + 临时文件 + `fsync` + `rename` 原子替换；
  向量以 base64 存原始 float 字节（1 万条 31MB），加载时兼容旧的 JSON 数组格式

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
  （对可能构成 URL 的内容另有跨事件判定），命中缓存的响应回放前同样要过一遍
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
| `cache.store_vectors` | true | 是否把向量一并落盘。`false` 时只存文本（原始提问已在 `src` 字段里），启动按它重算向量；落盘更小，代价是启动多一次全量编码（落在后台建图线程里） |
| `filter.max_input_chars` | 500 | 输入最大字符 |
| `filter.max_output_chars` | 0 | 输出最大字符（0 = 不截断）。流式响应按**事件**累计，到达上限后补发一个 `data: [DONE]` 并停止透传——截断也必须给客户端一个明确的流结束标志 |
| `filter.block_urls` | false | 是否拦截输出里的 URL。**默认关闭**：它只能拦"LLM 顺手贴出的链接"这类无意情形，对有意变形的 URL（`hxxp://`、插零宽字符、编码拼接）天然无效——那是模式匹配的边界；开着还会误伤正常回答里的文档链接与代码示例。需要"减少无意暴露面"时再打开 |
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

向量索引与降级单列一组：`ai_gateway_index_ready`（1 = 语义检索可用，0 = 后台建图中）、
`ai_gateway_index_building`（1 = 有后台建图在进行）、`ai_gateway_index_vectors`（当前索引中的向量数）、
`ai_gateway_cache_degraded_searches_total`（索引不可用时改走线性扫描的查询数）——
没有这几个量，"后台在忙"只能表现为命中率莫名下降。

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

两个测量脚本（README 里的性能数字由它们产出，原始输出在 `results/`）：
`scripts/latency_bench.py`（keep-alive 连接下的命中/未命中延迟分位数）、
`scripts/fill_cache_bench.py`（灌库 + 逐点采样 RSS：稳态标记、落盘窗口、峰值）。

两点判据上的说明：

- `test_pipeline` 链接的是与可执行文件**同一份**请求管道对象代码（`ai_gateway_pipeline`），
  覆盖的是生产路径本身，而不是测试里另写的一份逻辑。此前管道只存在于 `main.cpp`，
  于是 `drain`、请求合并的命名空间隔离、keep-alive 决策三处缺陷都落在零覆盖的代码层。
- `integration_pipeline_test.sh` 在**启动实例时**做一次监听者校验（端口 inode ↔ 进程 fd 比对），
  避免端口被残留进程占着、测试却连到别的实例上得出假结论；A–F 六段共用这一个实例。

## 实测性能

> 测试环境：2 vCPU / 1968 MB 的 VPS（**压测端与被测服务同机环回**，服务 `taskset -c 0`、压测端 `taskset -c 1`）
> 上游：DeepSeek v4-flash（公网 API）；Embedding：ONNX bge-small-zh-v1.5 INT8 (512d)，进程内推理
> 原始输出全部在 `results/`（文件名带日期），测量脚本为 `scripts/latency_bench.py`、`scripts/fill_cache_bench.py`、
> `scripts/benchmark.py`、`scripts/eval_semantic_cache.py`。引用这些数字时请连同环境一起引用。

### 语义匹配质量（公开数据集）

缓存判定的质量用公开数据集量，而不是自建的相似句集合——自造集的重复度会把命中率抬到不可信的高位。

| 判定策略（LCQMC 3000 对） | 召回 | 误命中 | F1 |
|---|---|---|---|
| 0.85 + 实体一致性否决（当前默认） | 87.80% | **29.59%** | **80.42%** |
| 0.80 纯阈值 | 88.55% | 30.38% | 80.53% |

LCQMC 的"相似"包含问答对关系，当作"同义改写"用会天然偏高，因此**误命中率**才是关键指标：
0.80 时近三成的"命中"是错的；0.85 叠加实体一致性否决（数字 / 大写缩略语 / 混合标识符存在
不对称差集即否决）后降到 29.59%。PAWS-X 上精确率在所有阈值下都只有 43% 左右，说明这个量级的
embedding 对"语序颠倒 / 主宾互换"这类深层改写无能为力——这也是需要实体否决的原因。
阈值扫描与逐条明细由 `scripts/eval_semantic_cache.py` 输出（`results/eval_semantic_cache_*.log`）。

### 延迟

| 指标 | 数值 | 说明 |
|---|---|---|
| 命中 p50 / p95（真实上游，两轮回放） | 12 ms / 21 ms | 含进程内向量化与图检索 |
| 未命中 p50（真实上游） | 608 – 669 ms | 含公网 LLM API 往返，衡量的是上游不是网关 |
| 命中 p50 / 未命中 p50（mock 上游） | **6.9 ms** / 8.0 ms | 本地 mock 上游，量的是**网关自身**处理开销 |

mock 那一行是刻意补的：把上游摘掉之后，命中 6.9 ms 就是"进程内编码 + 图检索 + 输出过滤 +
HTTP 处理"的全部开销；未命中 p95 明显高于 p50（48.8 ms vs 8.0 ms），说明满缓存下的淘汰与
分配抖动集中在少数请求上。

### 流式

| 指标 | 回源（未命中） | 命中（回放） |
|---|---|---|
| 端到端 | 1.421 s | **0.015 s** |
| 客户端收到的字节 | 36,790 | 36,803（= 首次 + 注释行，去掉注释行后**逐字节相同**） |

流式命中回放的是当初记下的**原始 SSE 字节**，因此客户端侧无需任何特殊处理：
用 OpenAI 兼容 SDK（`@earendil-works/pi-ai`）复测，两次的事件序列完全一致
（`thinking_*` → `text_start` → `text_delta` × 81 → `text_end` → `done`），usage 完整。

### 内存与落盘

向量以 **base64 原始 float 字节**落盘（而不是 JSON 数字数组）。这条选择决定了下面两列的差距：

| 状态 | 数值 |
|---|---|
| 空载 RSS | 62.6 MB |
| 灌入 5000 条 | 123.9 MB |
| 灌入 10000 条 | 178.0 MB |
| 静置后（含落盘） | 230.1 MB |
| 峰值 | 230.1 MB（与静置值相同——落盘不再抬高水位） |
| 落盘文件（10000 条） | **31.0 MB** |

对照：若向量按 JSON 数字数组写（每个 float 21 字符，因为序列化按 17 位有效数字输出），
同样 1 万条的落盘是 107.7 MB、峰值 373 MB —— 峰值那部分主要来自落盘期间要构建等量的 DOM
与 dump 字符串。换 base64 之后，DOM 里每个向量只是一个字符串。

### 启动

| 指标 | 数值 |
|---|---|
| 加载 10000 条缓存（31 MB）到**可接受连接** | **约 0.60 s**（5 轮实测中位 0.597 s、范围 0.574–0.662 s，见 `results/startup_bench_20260921.txt`；语义检索可用还需叠加后台建图时间，见下行） |
| 加载 10000 条（旧 108 MB 格式）到可服务 | 2.49 s |
| 后台建图（10000 条真实 embedding） | 约 7 s |

建图已移到后台线程，因此它**不阻塞服务**：进程在约 0.60 s 就能接受连接，建图期间
`/metrics` 的 `index_ready` 为 0，语义检索退化为线性扫描（详见下节），建图完成后自动恢复。

### 检索：索引与降级

| 规模 | 自检索 top-1 | top-3 召回 | HNSW 检索 | 暴力检索 | 加速比 |
|---|---|---|---|---|---|
| 真实集 320（ONNX 编码） | **200/200 = 100%** | 100% | 201.9 µs | 264.5 µs | 1.31x |
| 合成 1000 | 200/200 = 100% | 99.67% | 658.9 µs | 772.9 µs | 1.17x |
| 合成 5000 | 200/200 = 100% | 93.83% | 1.48 ms | 4.04 ms | 2.73x |
| 合成 10000 | 200/200 = 100% | 81.67% | 1.74 ms | 8.18 ms | 4.70x |

`ef_search=50` 固定时，规模增大会让束搜索覆盖不足（HNSW 固有特性）；缓存只取 top-1，
这一项不影响命中。**建图耗时对数据分布极敏感**：真实 embedding 1 万条约 7 s，
而 512 维**随机**向量（无簇结构、点与点近似等距，启发式邻居选择无从剪枝）要 117 s ——
评估建图性能必须用带簇结构的数据。

索引不可用时（启动或重建窗口）检索**不退化为"放弃语义"**，而是走 `LruStore::scan_topk()`
线性扫描：建图窗口内命中 p50 **14.8 ms**（同一时刻 HNSW 路径 7.2 ms），50/50 全部命中。
代价约 7.6 ms，换来的是这段时间不必回源（未命中一次 669 ms 且真花 token）。
走了降级路径的查询数会在 `/metrics` 的 `ai_gateway_cache_degraded_searches_total` 里单列。

### 真实上游回放

824 次真实请求（DeepSeek、阈值 0.85 + 实体否决、空缓存起步）：

| 数据集 | R1 | R2 | 合计 | 命中 p50 | 未命中 p50 |
|---|---|---|---|---|---|
| synthetic 320 × 2 轮 | 16.2% | 100% | **58.1%** | 12 ms | 669 ms |
| real 92 × 2 轮 | 5.4% | 100% | **52.7%** | 16 ms | 608 ms |

`/metrics` 汇总：命中 469 / 未命中 355 = **56.9%**，省下 **30,290 tokens**。
注意 `tokens_saved` 是按已发生未命中的平均调用量外推的**估算值**，对外引用用 token 数比
用金额稳（金额随上游定价变动）。R2 的 100% 里很大一部分是"命中自己上一轮写入的向量"，
所以自建集的合计命中率只作链路是否打通的证据，不当生产命中率引用。

