// LRU + TTL 缓存存储实现
#include "lru_store.h"

#include <algorithm>
#include <fstream>
#include <functional>

#include <nlohmann/json.hpp>

namespace ai_gateway {

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

void LruStore::put(const std::string& key, std::string value) {
  put_with_embedding(key, std::move(value), {});
}

void LruStore::put_with_embedding(const std::string& key,
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
  node.key = key;
  node.value = std::move(value);
  node.embedding.data = std::move(embedding);
  node.ctime = Clock::now();
  lru_.push_front(std::move(node));
  iter_map_[key] = lru_.begin();
}

std::vector<float> LruStore::get_embedding(const std::string& key) {
  std::lock_guard lock(mutex_);
  auto it = iter_map_.find(key);
  if (it == iter_map_.end()) return {};
  return it->second->embedding.data;
}

void LruStore::for_each_embedding(
    const std::function<void(const std::string&,
                             const std::vector<float>&)>& fn) const {
  std::lock_guard lock(mutex_);
  auto now = Clock::now();
  for (const auto& node : lru_) {
    // 跳过过期条目
    if (ttl_seconds_ > 0) {
      auto age = std::chrono::duration_cast<std::chrono::seconds>(
          now - node.ctime).count();
      if (age >= ttl_seconds_) continue;
    }
    if (!node.embedding.data.empty()) {
      fn(node.key, node.embedding.data);
    }
  }
}

size_t LruStore::size() const {
  std::lock_guard lock(mutex_);
  return lru_.size();
}

void LruStore::expire_one(const std::string& key) {
  // 调用者已持有 mutex 不需要重复上锁
  auto it = iter_map_.find(key);
  if (it == iter_map_.end()) return;
  lru_.erase(it->second);
  iter_map_.erase(it);
}

bool LruStore::save(const std::string& path) const {
  std::lock_guard lock(mutex_);
  try {
    nlohmann::json arr = nlohmann::json::array();
    auto now = Clock::now();
    for (const auto& node : lru_) {
      // 跳过过期条目
      if (ttl_seconds_ > 0) {
        auto age = std::chrono::duration_cast<std::chrono::seconds>(
            now - node.ctime).count();
        if (age >= ttl_seconds_) continue;
      }
      nlohmann::json entry;
      entry["key"] = node.key;
      entry["value"] = node.value;
      if (!node.embedding.data.empty())
        entry["embedding"] = node.embedding.data;
      entry["ctime"] = std::chrono::duration_cast<std::chrono::seconds>(
          node.ctime.time_since_epoch()).count();
      arr.push_back(std::move(entry));
    }
    std::ofstream ofs(path);
    ofs << arr.dump();
    return true;
  } catch (...) { return false; }
}

bool LruStore::load(const std::string& path) {
  std::lock_guard lock(mutex_);
  try {
    std::ifstream ifs(path);
    if (!ifs) return false;
    auto arr = nlohmann::json::parse(ifs);

    lru_.clear();
    iter_map_.clear();

    for (auto& entry : arr) {
      Node node;
      node.key = entry["key"].get<std::string>();
      node.value = entry["value"].get<std::string>();
      if (entry.contains("embedding")) {
        node.embedding.data = entry["embedding"].get<std::vector<float>>();
      }
      auto secs = entry["ctime"].get<int64_t>();
      node.ctime = Clock::time_point(std::chrono::seconds(secs));
      lru_.push_back(std::move(node));
      iter_map_[lru_.back().key] = --lru_.end();
    }
    return true;
  } catch (...) { return false; }
}

}  // namespace ai_gateway
