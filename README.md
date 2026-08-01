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
- **向量检索**: 简化实现的 HNSW 图索引（未实现启发剪枝），10K 向量内召回率 95%
- **存储引擎**: LRU + TTL 缓存管理，JSON 持久化
- **嵌入推理**: C++ ONNX Runtime 进程内 INT8 量化推理，零外部依赖
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
- **HnswIndex**: HNSW 图索引（header-only，简化实现）
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
                                          test_stats

for t in build/tests/test_*; do $t; done
# 6/6 模块, 36/36 用例
```

## 实测性能

> 测试环境：Ubuntu 24.04, 2GB VPS, 1 vCPU, GCC 13.3 (-O2 -mavx2 -mfma)
> 后端：DeepSeek v4-flash, Embedding：ONNX bge-small-zh-v1.5 INT8 (512d)
> 测试数据：94 条中文用户消息 × 2 轮回放（10 语义簇）

| 指标 | 数值 | 说明 |
|------|------|------|
| 缓存命中率 | 89.9% | R1 84/94 + R2 85/94 (169/188) |
| 命中延迟 | ~241ms | C++ ONNX 推理 + HNSW 检索 |
| 未命中延迟 | ~1.3s | LLM API 端到端 |
| 加速比 | 5.4x | 命中 vs 未命中 |
| 费用节省 | ¥0.0378 | 430 请求累计节省 |
| 二进制大小 | ~2.2MB | release build |
| 网关 RSS | ~40MB | 含 INT8 ONNX 模型 (23MB) |
| 代理延迟 | < 1ms | 不含 LLM/Embedding |

