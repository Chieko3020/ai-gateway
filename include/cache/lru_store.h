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

  // 记录"该条目对应的原始查表键"（namespace:user_message）。
  // 语义缓存的条目键是 msg:N，向量检索失败时的降级路径需要原样文本才能精确命中，
  // 因此把来源键挂在同一条目上（旧实现为此额外存了一份完整回复，见报告 M4）
  void set_source(const std::string& key, std::string source);

  // 精确匹配：key 命中或条目的 source 命中，返回第一条未过期条目的回复。
  // 用于 embedding 不可用时的降级路径（旧实现靠"再存一份 ns_key"实现）
  std::optional<std::string> get_exact(const std::string& lookup_key);

  // 遍历所有非过期条目的 embedding（供索引重建使用）
  void for_each_embedding(
      const std::function<void(const std::string&,
                               const std::vector<float>&)>& fn) const;

  // 主动清理所有过期条目，返回清理数量
  // 用途：定时维护与持久化前调用——避免失效条目长期占用内存，
  //       并让 size() 与落盘文件只反映仍然有效的条目
  size_t purge_expired();

  // 持久化：保存到 JSON 文件。
  // 实现为"锁内取快照 + 锁外序列化 + 临时文件 fsync + rename 原子替换"：
  // 锁内做 DOM 构建与写盘会让所有 get/put 停顿数秒（报告 M2），
  // 直接截断写原文件则在崩溃时留下半截 JSON、下次 load 失败导致整份缓存丢失（M9）
  bool save(const std::string& path) const;

  // 持久化：从 JSON 文件加载
  bool load(const std::string& path);

  // ---- embedding 指纹（本轮新增）-----------------------------------------
  // 落盘时写入、加载时比对：向量只对"产生它的模型"有意义，换模型/改分词之后
  // 新旧向量不在同一个空间里，静默命中会返回错误答案。见 embedding_fingerprint.h
  //
  // 必须在 load() 之前设置：load() 会用给定指纹校验文件内容
  void set_fingerprint(std::string fp) { expected_fingerprint_ = std::move(fp); }
  const std::string& fingerprint() const { return fingerprint_; }

  // 文件里记录的指纹（load 后有效；空串 = 旧格式文件没有该字段）
  const std::string& loaded_fingerprint() const { return fingerprint_; }
  bool loaded_file_had_fingerprint() const { return file_had_fingerprint_; }
  // 因指纹不匹配而被丢弃的向量条数（load 后有效）
  size_t dropped_vectors() const { return dropped_vectors_; }
  // 本次 load 是否因为指纹不一致而丢弃了向量（调用方据此决定是否整份丢弃）
  bool fingerprint_mismatch() const { return fingerprint_mismatch_; }

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
    std::string source;  // 原始查表键（namespace:user_message），可为空
    Embedding embedding;
    TimePoint ctime;  // 创建时间（TTL 判断用）
  };

  // save() 用的条目快照：在锁内只做"拷贝数据"，锁外才是 DOM/写盘
  struct SaveEntry {
    std::string key;
    std::string value;
    std::string source;
    std::vector<float> embedding;
    int64_t ctime_seconds = 0;
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

  // 期望的 embedding 指纹（调用方通过 set_fingerprint 给出）
  std::string expected_fingerprint_;
  // 文件里实际记录的指纹（空 = 无该字段）
  std::string fingerprint_;
  bool file_had_fingerprint_ = false;
  size_t dropped_vectors_ = 0;
  // 是否发生过指纹不一致（调用方据此决定"丢弃整个文件"还是"保留文本"）
  bool fingerprint_mismatch_ = false;
};

}  // namespace ai_gateway
