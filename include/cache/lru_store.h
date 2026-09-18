// LRU + TTL 内存缓存存储：O(1) 查找 + 有序淘汰 + 线程安全
//
// 内部结构：
//   std::list<Node> ，双向链表 头部是最近使用 尾部最久未用
//   std::unordered_map<key, iterator> ，O(1) 定位链表节点
//
// 线程安全：所有 public 方法持有 mutex，适合低并发场景
#pragma once

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <list>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace ai_gateway {

class LruStore {
 public:
  // 存储的向量（一个缓存条目可能关联一个 embedding 向量）
  struct Embedding {
    std::vector<float> data;
  };

  // max_entries=0 表示不限制数量；ttl_seconds=0 表示永不过期
  explicit LruStore(size_t max_entries = 10000, int64_t ttl_seconds = 604800);

  // 查找缓存：命中时移动到 LRU 头部并返回值；过期/未命中返回 nullopt
  std::optional<std::string> get(const std::string& key);

  // 存入缓存（不含 embedding）：
  //   key 已存在 移动到头部 + 更新 value
  //   key 不存在 插入头部；若超出 max_entries 则淘汰尾部
  void put(std::string key, std::string value);

  // 存入缓存 + embedding 向量
  void put_with_embedding(std::string key,
                          std::string value,
                          std::vector<float> embedding);

  // 获取关联的 embedding；不存在返回空 vector
  std::vector<float> get_embedding(const std::string& key);

  // 遍历所有非过期条目的 embedding（供索引重建使用）
  void for_each_embedding(
      const std::function<void(const std::string&,
                               const std::vector<float>&)>& fn) const;

  // 主动清理所有过期条目，返回清理数量
  // 用途：定时维护与持久化前调用——避免失效条目长期占用内存，
  //       并让 size() 与落盘文件只反映仍然有效的条目
  size_t purge_expired();

  // 持久化：保存到 JSON 文件
  bool save(const std::string& path) const;

  // 持久化：从 JSON 文件加载
  bool load(const std::string& path);

  // 统计
  size_t size() const;
  size_t hit_count() const { return hit_count_; }
  size_t miss_count() const { return miss_count_; }
  size_t evict_count() const { return evict_count_; }
  size_t expired_count() const { return expired_count_; }
  size_t max_entries() const { return max_entries_; }
  int64_t ttl_seconds() const { return ttl_seconds_; }

 private:
  // 淘汰过期条目（get 未命中时调用）
  void expire_one(const std::string& key);

  using Clock = std::chrono::system_clock;      // 系统时钟 （epoch 持久化）

  using TimePoint = Clock::time_point;

  struct Node {
    std::string key;
    std::string value;
    Embedding embedding;
    TimePoint ctime;  // 创建时间（TTL 判断用）
  };

  using LruList = std::list<Node>;
  using IterMap = std::unordered_map<std::string, LruList::iterator>;

  size_t max_entries_;
  int64_t ttl_seconds_;

  LruList lru_;
  IterMap iter_map_;

  mutable std::mutex mutex_;

  size_t hit_count_ = 0;
  size_t miss_count_ = 0;
  size_t evict_count_ = 0;
  size_t expired_count_ = 0;
};

}  // namespace ai_gateway
