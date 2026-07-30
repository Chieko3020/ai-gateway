# AI Gateway — C++20 语义缓存透明代理

- 部署在客户端与 LLM API 之间，通过语义缓存减少重复的 API 调用节省 token
- epoll ET + 非阻塞 I/O + 单 Reactor + 线程池架构，适配并发场景
- 自研向量索引，AVX2 SIMD 加速内积计算，零外部向量库依赖
- 本地 ONNX Runtime 运行 bge-small-zh 模型，也可调用模型提供商的 Embedding 模型
- LLM 后端和 Embedding 后端独立配置，支持任意 OpenAI 兼容 API
- LRU + TTL 内存缓存 + JSON 持久化，重启不丢缓存，默认 7 天自动过期
- 开发环境：Ubuntu 24.04 LTS, GCC 13.3, CMake 3.28

## 项目简介

客户端每日处理大量用户消息，其中语义重复或相似的请求反复调用 LLM API，造成不必要的费用。AI Gateway 作为透明代理层插入客户端与 LLM 后端之间，通过 Embedding 向量化用户消息、AVX2 SIMD 检索历史缓存、余弦相似度判断是否命中，命中时直接返回缓存，未命中时转发 API 处理消息并将结果写入缓存。相较于现有方案（GPTCache / LiteLLM），本项目的 C++ 实现更为轻量

## 功能特性

### 核心功能
- **语义缓存**: 基于 bge-small-zh Embedding 的向量检索 + 余弦相似度匹配，非精确文本也能命中
- **精确降级**: Embedding 服务不可用时自动降级为精确匹配，保证服务可用
- **安全过滤**: 请求/响应双向过滤（反注入检测、URL 拦截、关键词屏蔽、长度截断）
- **实时统计**: 每 60 秒输出命中率/token 消耗/费用节省/延迟统计到 systemd journal

### 技术特性
- **并发模型**: 单 Reactor + 工作线程池，主线程负责连接管理，线程池处理缓存查询和 LLM 转发
- **向量检索**: 暴力搜索 + AVX2 SIMD
- **存储引擎**: LRU + TTL 缓存管理，JSON 持久化

## 技术栈

| 组件 | 选型 |
|------|------|
| 编程语言 | C++20 |
| JSON | nlohmann/json |
| HTTP 客户端 | libcurl |
| Embedding 推理 | ONNX Runtime + bge-small-zh-v1.5 (512d) |
| 构建系统 | CMake 3.16+ |

## 项目结构

```
ai-gateway/
├── include/                     # 头文件
│   ├── http_server.h            #   epoll ET + 线程池
│   ├── connection_handler.h     #   请求处理：解析+路由+处理+响应
│   ├── filter.h                 #   安全过滤器
│   ├── cache_engine.h           #   缓存协调器
│   ├── vector_index.h           #   向量索引 + AVX2 SIMD
│   ├── lru_store.h              #   LRU + TTL 缓存存储
│   ├── embedding.h              #   Embedding API 客户端
│   ├── llm_client.h             #   LLM 后端转发
│   ├── stats.h                  #   统计模块
│   ├── config.h                 #   JSON 配置加载
│   ├── curl_client.h            #   RAII libcurl 封装
│   ├── thread_pool.h            #   线程池
│   └── logger.h                 #   日志宏
├── src/                         # 源文件
│   ├── main.cpp                 #   入口：组装 + 启动 + 信号处理
│   ├── server/                  #   模块1: HTTP 接入 + 过滤
│   ├── cache/                   #   模块2: 语义缓存
│   ├── backend/                 #   模块3: LLM 后端转发
│   ├── stats/                   #   模块4: 统计
│   └── common/                  #   公共组件（config/curl_client）
├── tests/                       # 单元测试
├── config/
│   ├── gateway.example.json     #   配置模板
│   └── gateway.env              #   API Key（systemd 注入）
├── scripts/
│   ├── ai-gateway.service       #   systemd 服务
│   ├── embed_server.service     #   ONNX 嵌入服务
│   └── benchmark.sh             #   压测脚本
└── CMakeLists.txt
```

### 核心模块

#### server — HTTP 接入层
- **HttpServer**: epoll ET 主循环和线程池
- **ConnectionHandler**: 请求处理
- **Filter**: 安全过滤（反注入/URL/关键词/长度）
- **Router**: URL 路由表
- **Request/Response**: HTTP/1.1

#### cache — 语义缓存引擎
- **CacheEngine**: 编排缓存流程（Embedding搜索，判断命中/未命中）
- **VectorIndex**: 向量索引 + AVX2 SIMD 内积 + Top-K 搜索
- **LruStore**: LRU + TTL 内存缓存 + JSON 持久化
- **Embedding**: OpenAI 兼容 Embedding API 客户端

#### backend — LLM 转发
- **LlmClient**: LLM API 客户端

#### stats — 统计
- **Stats**: 命中率/token/费用/延迟统计 + 定期输出

## 架构设计

```
客户端 (QQ Bot / Web / API)
    │  POST /v1/chat/completions
    ▼
┌─────────────────────────────────────────────────────────────────┐
│  HTTP 接入层 (epoll ET + 非阻塞 IO)                              │
│  → 模块1: ConnectionHandler (解析/路由/过滤)                     │
├─────────────────────────────────────────────────────────────────┤
│  安全过滤器 (MessageFilter)                                      │
│  → 模块1: 反注入/URL/关键词/长度 (双向)                           │
├─────────────────────────────────────────────────────────────────┤
│  语义缓存                                                        │
│  → 模块2: CacheEngine                                           │
│    1. Embedding API → 向量化用户消息 (ONNX, 512d)                │
│    2. VectorIndex.search → Top-K 相似条目 (AVX2 SIMD)           │
│    3. max(similarity) ≥ 0.85 (可更改) ？                        │
│       → HIT: 返回 LruStore 中的缓存回复 (总计 18ms)              │
│       → MISS: 继续 ↓                                            │
├─────────────────────────────────────────────────────────────────┤
│  LLM 后端转发                                                    │
│  → 模块3: LlmClient → OpenAI 兼容 API                            │
│  → 模块4: Stats (命中率/token/费用/延迟)                          │
└─────────────────────────────────────────────────────────────────┘
    │  ↑
    │  缓存写入（MISS 后）
    ▼
┌──────────────────────┐    ┌──────────────────────┐
│  LLM 后端 (可替换)    │    │  Embedding 后端 (可替换) │
│  DeepSeek / OpenAI /  │    │  ONNX bge-small-zh /   │
│  Ollama / vLLM / ...  │    │  OpenAI / 硅基流动 / ...│
└──────────────────────┘    └──────────────────────┘
```

## 编译和运行

### 依赖

```bash
sudo apt install build-essential cmake g++-13 \
    libcurl4-openssl-dev nlohmann-json3-dev
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
# Embedding 服务（首次需安装 ONNX）
/tmp/onnx-venv/bin/pip install onnxruntime numpy flask
/tmp/onnx-venv/bin/python scripts/embed_server.py &

# 网关
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
sudo cp scripts/ai-gateway.service scripts/embed_server.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now ai-gateway embed_server
```

## 配置项

| 字段 | 默认 | 说明 |
|------|------|------|
| `server.port` | 3003 | 监听端口 |
| `backend.url` | DeepSeek API | LLM 后端 URL（OpenAI 兼容） |
| `backend.model` | deepseek-v4-flash | 模型名称 |
| `backend.timeout_seconds` | 60 | 请求超时 |
| `embedding.url` | 127.0.0.1:8081 | Embedding 服务地址 |
| `cache.enabled` | true | 启用语义缓存 |
| `cache.similarity_threshold` | 0.85 | 余弦相似度阈值 |
| `cache.max_entries` | 10000 | 最大缓存条目 |
| `cache.ttl_days` | 7 | 缓存过期天数 |
| `filter.max_input_chars` | 500 | 输入最大字符 |
| `filter.max_output_chars` | 600 | 输出最大字符 |
| `filter.block_urls` | true | 拦截 URL |

API Key 通过 `config/gateway.env`（systemd `EnvironmentFile`）或环境变量 `LLM_API_KEY` 注入。

## 测试

```bash
cmake --build build --target test_filter test_request test_response test_router \
                                          test_lru_store test_vector_index \
                                          test_stats test_config

for t in build/tests/test_*; do $t; done
# 8/8 模块, 43/43 用例
```

## 实测性能

> 示例测试环境，实际结果因环境和用例不同会有所差异
> 测试环境：Ubuntu 24.04, 2GB VPS, 1 vCPU, GCC 13.3 (-O2 -mavx2 -mfma)
> 后端：DeepSeek v4-flash, Embedding：ONNX bge-small-zh-v1.5 (512d)
> 测试数据：94 条中文用户消息，覆盖问候/天气/编程/AI/数学/生活/翻译/常识等常见的用户对话语义簇

| 指标 | 数值 | 说明 |
|------|------|------|
| 缓存命中率 | 70.2% | 94 请求，66 命中（含语义变体） |
| 命中延迟 | 18ms | Embedding 15ms + 向量检索 3ms |
| 未命中延迟 | ~2.7s | LLM API 端到端 |
| 加速比 | 153x | 命中 vs 未命中 |
| 并发加速 | 3.8x | 5 并发请求，4 线程池 |
| 费用节省 | 2.3x | ¥0.0757 节省 vs ¥0.0324 消耗 |
| 二进制大小 | 560KB | stripped |
| 网关 RSS | 15MB | ONNX 服务 71MB 独立 |
| 代理延迟 | < 1ms | 不含 LLM/Embedding |
