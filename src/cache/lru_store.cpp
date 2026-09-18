// LRU + TTL 缓存存储实现
#include "cache/lru_store.h"

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <cstdio>
#include <fstream>
#include <functional>

#include <nlohmann/json.hpp>

namespace ai_gateway {

// 落盘文件里承载元数据的保留键。用 "__" 前后缀是为了不与业务键冲突：
// 业务键形如 "[ns:]msg:N"（见 cache_engine.cpp），不会出现双下划线前后缀
constexpr const char* kMetaKey = "__meta__";

LruStore::LruStore(size_t max_entries, int64_t ttl_seconds)
    : max_entries_(max_entries), ttl_seconds_(ttl_seconds) {}

std::optional<std::string> LruStore::get(const std::string& key) {
  std::lock_guard lock(mutex_);

  auto it = iter_map_.find(key);
  if (it == iter_map_.end()) {
    ++miss_count_;
    return std::nullopt;
  }

  auto& node = *(it->second);

  // TTL 过期检查
  if (ttl_seconds_ > 0) {
    auto age = std::chrono::duration_cast<std::chrono::seconds>(
        Clock::now() - node.ctime).count();
    if (age >= ttl_seconds_) {
      expire_one(key);
      ++expired_count_;
      ++miss_count_;
      return std::nullopt;
    }
  }

  // 命中则移动到 LRU 头部
  lru_.splice(lru_.begin(), lru_, it->second);
  ++hit_count_;
  return node.value;
}

void LruStore::put(std::string key, std::string value) {
  put_with_embedding(std::move(key), std::move(value), {});
}

void LruStore::put_with_embedding(std::string key,
                                   std::string value,
                                   std::vector<float> embedding) {
  std::lock_guard lock(mutex_);

  auto it = iter_map_.find(key);
  if (it != iter_map_.end()) {
    // 已存在 更新并移到头部
    auto& node = *(it->second);
    node.value = std::move(value);
    node.embedding.data = std::move(embedding);
    node.ctime = Clock::now();
    lru_.splice(lru_.begin(), lru_, it->second);
    return;
  }

  // LRU淘汰 超出限制时淘汰尾部（最久未用）
  if (max_entries_ > 0 && lru_.size() >= max_entries_) {
    auto& back = lru_.back();
    iter_map_.erase(back.key);
    lru_.pop_back();
    ++evict_count_;
  }

  // 插入头部
  Node node;
  node.key = std::move(key);
  node.value = std::move(value);
  node.embedding.data = std::move(embedding);
  node.ctime = Clock::now();
  lru_.push_front(std::move(node));
  iter_map_[lru_.front().key] = lru_.begin();
}

std::vector<float> LruStore::get_embedding(const std::string& key) {
  std::lock_guard lock(mutex_);
  auto it = iter_map_.find(key);
  if (it == iter_map_.end()) return {};

  auto& node = *(it->second);
  if (ttl_seconds_ > 0) {
    auto age = std::chrono::duration_cast<std::chrono::seconds>(
        Clock::now() - node.ctime).count();
    if (age >= ttl_seconds_) return {};
  }
  return node.embedding.data;
}

void LruStore::for_each_embedding(
    const std::function<void(const std::string&,
                             const std::vector<float>&)>& fn) const {
  // 先在锁内把 (key, embedding) 拷出来，再在锁外回调：
  // 回调里做的是 HNSW 建图（每个条目一次 O(ef_construction) 搜索），
  // 在持锁状态下跑会让整段重建期间所有 get/put 阻塞（报告 M2）
  std::vector<std::pair<std::string, std::vector<float>>> snapshot;
  {
    std::lock_guard lock(mutex_);
    auto now = Clock::now();
    snapshot.reserve(lru_.size());
    for (const auto& node : lru_) {
      // 跳过过期条目
      if (ttl_seconds_ > 0) {
        auto age = std::chrono::duration_cast<std::chrono::seconds>(
            now - node.ctime).count();
        if (age >= ttl_seconds_) continue;
      }
      if (!node.embedding.data.empty())
        snapshot.emplace_back(node.key, node.embedding.data);
    }
  }
  for (const auto& [key, emb] : snapshot) fn(key, emb);
}

void LruStore::set_source(const std::string& key, std::string source) {
  std::lock_guard lock(mutex_);
  auto it = iter_map_.find(key);
  if (it == iter_map_.end()) return;
  it->second->source = std::move(source);
}

std::optional<std::string> LruStore::source_of(const std::string& key) const {
  std::lock_guard lock(mutex_);
  auto it = iter_map_.find(key);
  if (it == iter_map_.end()) return std::nullopt;
  // 过期条目按不存在处理（与 get() 的口径一致，只是这里不真删——
  // 只读方法不该改容器；真正清理由 get()/purge_expired() 负责）
  if (ttl_seconds_ > 0) {
    auto age = std::chrono::duration_cast<std::chrono::seconds>(
        Clock::now() - it->second->ctime).count();
    if (age >= ttl_seconds_) return std::nullopt;
  }
  if (it->second->source.empty()) return std::nullopt;
  return it->second->source;
}

std::optional<std::string> LruStore::get_exact(const std::string& lookup_key) {
  std::lock_guard lock(mutex_);
  auto now = Clock::now();
  for (auto it = lru_.begin(); it != lru_.end(); ++it) {
    const bool matches = (it->key == lookup_key) || (it->source == lookup_key);
    if (!matches) continue;
    if (ttl_seconds_ > 0) {
      auto age = std::chrono::duration_cast<std::chrono::seconds>(
          now - it->ctime).count();
      if (age >= ttl_seconds_) continue;
    }
    lru_.splice(lru_.begin(), lru_, it);
    ++hit_count_;
    return it->value;
  }
  ++miss_count_;
  return std::nullopt;
}

size_t LruStore::size() const {
  std::lock_guard lock(mutex_);
  return lru_.size();
}

size_t LruStore::purge_expired() {
  std::lock_guard lock(mutex_);
  if (ttl_seconds_ <= 0) return 0;  // 0 表示永不过期

  auto now = Clock::now();
  size_t purged = 0;
  for (auto it = lru_.begin(); it != lru_.end();) {
    auto age = std::chrono::duration_cast<std::chrono::seconds>(
        now - it->ctime).count();
    if (age >= ttl_seconds_) {
      iter_map_.erase(it->key);
      it = lru_.erase(it);
      ++purged;
      ++expired_count_;
    } else {
      ++it;
    }
  }
  return purged;
}

void LruStore::expire_one(const std::string& key) {
  // 调用者已持有 mutex 不需要重复上锁
  auto it = iter_map_.find(key);
  if (it == iter_map_.end()) return;
  lru_.erase(it->second);
  iter_map_.erase(it);
}

bool LruStore::save(const std::string& path) const {
  // ---- 1. 锁内只取快照（拷贝），不建 DOM、不写盘 ----
  std::vector<SaveEntry> snapshot;
  {
    std::lock_guard lock(mutex_);
    auto now = Clock::now();
    snapshot.reserve(lru_.size());
    for (const auto& node : lru_) {
      // 跳过过期条目
      if (ttl_seconds_ > 0) {
        auto age = std::chrono::duration_cast<std::chrono::seconds>(
            now - node.ctime).count();
        if (age >= ttl_seconds_) continue;
      }
      SaveEntry e;
      e.key = node.key;
      e.value = node.value;
      e.source = node.source;
      e.embedding = node.embedding.data;
      e.ctime_seconds = std::chrono::duration_cast<std::chrono::seconds>(
                            node.ctime.time_since_epoch())
                            .count();
      snapshot.push_back(std::move(e));
    }
  }

  // ---- 2. 锁外序列化 ----
  std::string payload;
  try {
    nlohmann::json arr = nlohmann::json::array();
    // 指纹元数据条目：key 固定为 "__meta__"。用"数组里的第一个条目"而不是
    // 改成对象格式（{"meta":..., "entries":[...]}），是为了让**旧版本程序**
    // 读新文件时仍按数组解析——它只会多出一条 value 为空的条目，不会解析失败
    if (!expected_fingerprint_.empty()) {
      nlohmann::json meta;
      meta["key"] = kMetaKey;
      meta["fp"] = expected_fingerprint_;
      arr.push_back(std::move(meta));
    }
    for (auto& e : snapshot) {
      nlohmann::json entry;
      entry["key"] = e.key;
      entry["value"] = e.value;
      // src: 原始查表键（旧文件没有这个字段，load 时按空处理）
      if (!e.source.empty()) entry["src"] = e.source;
      if (!e.embedding.empty()) entry["embedding"] = e.embedding;
      entry["ctime"] = e.ctime_seconds;
      arr.push_back(std::move(entry));
    }
    payload = arr.dump();
  } catch (...) {
    return false;
  }

  // ---- 3. 临时文件 + fsync + rename 原子替换 ----
  // 直接写目标文件会先截断它：写到一半崩溃就留下半截 JSON，
  // 下次 load() 解析失败 → 整份缓存丢失（报告 M9）
  const std::string tmp = path + ".tmp";
  {
    // 父目录不存在时主动补建：旧实现直接 open 失败且返回 true，
    // 调用方以为"已持久化"，实际一份缓存都没落盘（报告 M9）
    auto slash = path.find_last_of('/');
    if (slash != std::string::npos && slash > 0 &&
        ::access(path.substr(0, slash).c_str(), F_OK) != 0) {
      if (::mkdir(path.substr(0, slash).c_str(), 0755) != 0 &&
          errno != EEXIST)
        return false;
    }
    std::ofstream ofs(tmp, std::ios::binary | std::ios::trunc);
    if (!ofs) return false;
    ofs << payload;
    ofs.flush();
    if (!ofs) return false;
  }
  int fd = ::open(tmp.c_str(), O_RDONLY);
  if (fd >= 0) {
    ::fsync(fd);
    ::close(fd);
  } else {
    return false;
  }
  if (::rename(tmp.c_str(), path.c_str()) != 0) {
    ::unlink(tmp.c_str());
    return false;
  }
  return true;
}

bool LruStore::load(const std::string& path) {
  std::lock_guard lock(mutex_);
  try {
    std::ifstream ifs(path);
    if (!ifs) return false;
    auto arr = nlohmann::json::parse(ifs);

    lru_.clear();
    iter_map_.clear();
    hit_count_ = 0;
    miss_count_ = 0;
    evict_count_ = 0;
    expired_count_ = 0;
    fingerprint_.clear();
    file_had_fingerprint_ = false;
    dropped_vectors_ = 0;
    fingerprint_mismatch_ = false;

    // ---- 1. 先取指纹元数据 ----
    for (auto& entry : arr) {
      if (!entry.is_object()) continue;
      if (entry.value("key", std::string{}) != kMetaKey) continue;
      if (entry.contains("fp") && entry["fp"].is_string()) {
        fingerprint_ = entry["fp"].get<std::string>();
        file_had_fingerprint_ = true;
      }
      break;
    }

    // ---- 2. 判定是否丢弃向量 ----
    //   调用方给了期望指纹（有模型）时：
    //     - 文件里没有指纹（旧格式）      -> 丢弃向量（来源不可考，见简报说明）
    //     - 指纹不一致（换模型/改分词）    -> 丢弃向量
    //   调用方没给期望指纹（无模型/降级模式）-> 不做校验，原样加载
    const bool check = !expected_fingerprint_.empty();
    const bool mismatch =
        check && (!file_had_fingerprint_ || fingerprint_ != expected_fingerprint_);
    fingerprint_mismatch_ = mismatch;  // fingerprint_ 保留文件里的原值供日志/诊断

    // ---- 3. 逐条加载 ----
    for (auto& entry : arr) {
      if (!entry.is_object()) continue;
      auto key_it = entry.find("key");
      if (key_it == entry.end() || !key_it->is_string()) continue;
      if (*key_it == kMetaKey) continue;  // 元数据条目不是缓存内容
      auto val_it = entry.find("value");
      if (val_it == entry.end() || !val_it->is_string())
        continue;  // 旧程序读新文件时会见到没有 value 的元数据条目，跳过即可

      Node node;
      node.key = key_it->get<std::string>();
      node.value = val_it->get<std::string>();
      // 向后兼容：src 是本轮新增字段，旧落盘文件没有
      if (entry.contains("src") && entry["src"].is_string())
        node.source = entry["src"].get<std::string>();
      if (entry.contains("embedding")) {
        auto vec = entry["embedding"].get<std::vector<float>>();
        // 指纹不一致：**丢弃向量、保留文本**。
        // 为什么不整份丢弃：文本条目仍能通过 source 精确命中（embedding 不可用
        // 时的降级路径本来就是这个语义），而语义索引可以按新模型重建；
        // 真正危险的是"拿旧向量当新向量用"，这一条被严格禁止
        if (mismatch) {
          ++dropped_vectors_;
        } else if (!vec.empty()) {
          node.embedding.data = std::move(vec);
        }
      }
      auto ct = entry.find("ctime");
      auto secs = (ct != entry.end() && ct->is_number_integer())
                      ? ct->get<int64_t>()
                      : 0;
      node.ctime = Clock::time_point(std::chrono::seconds(secs));
      lru_.push_back(std::move(node));
      iter_map_[lru_.back().key] = --lru_.end();
    }
    return true;
  } catch (...) { return false; }
}

}  // namespace ai_gateway
