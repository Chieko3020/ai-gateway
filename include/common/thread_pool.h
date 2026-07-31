// 固定大小线程池：任务队列 + 工作线程，析构时自动 join
#pragma once

#include <atomic>
#include <condition_variable>
#include <functional>
#include <future>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

namespace ai_gateway {

class ThreadPool {
 public:
  explicit ThreadPool(size_t num_threads = 0) {
    if (num_threads == 0) {
      // 线程数为硬件cpu并发数，如果无法获取则默认为 2
      num_threads = std::thread::hardware_concurrency();
      if (num_threads == 0) num_threads = 2;
    }
    for (size_t i = 0; i < num_threads; ++i) {
      workers_.emplace_back([this] { worker_loop(); });
    }
  }

  ~ThreadPool() {
    {
      std::lock_guard lock(mutex_);
      stop_ = true;
    }
    cv_.notify_all();
    for (auto& t : workers_)
      if (t.joinable()) t.join();
  }

  ThreadPool(const ThreadPool&) = delete;
  ThreadPool& operator=(const ThreadPool&) = delete;

  // 提交任务，返回 future 以获取结果
  template <typename F, typename... Args>
  auto submit(F&& f, Args&&... args)
      -> std::future<std::invoke_result_t<F, Args...>> {
    using Ret = std::invoke_result_t<F, Args...>;
    auto task = std::make_shared<std::packaged_task<Ret()>>(
        std::bind(std::forward<F>(f), std::forward<Args>(args)...));
    auto fut = task->get_future();
    {
      std::lock_guard lock(mutex_);
      if (stop_) throw std::runtime_error("submit on stopped ThreadPool");
      tasks_.emplace([task] { (*task)(); });
    }
    cv_.notify_one();
    return fut;
  }

  // 提交无返回值任务
  // 如果线程池已停止（析构中），任务仍会入队由析构函数排空，避免 fd 泄漏
  void execute(std::function<void()> task) {
    {
      std::lock_guard lock(mutex_);
      tasks_.emplace(std::move(task));
    }
    cv_.notify_one();
  }

  size_t size() const { return workers_.size(); }
  size_t pending() const {
    std::lock_guard lock(mutex_);
    return tasks_.size();
  }

 private:
  void worker_loop() {
    while (true) {
      std::function<void()> task;
      {
        std::unique_lock lock(mutex_);
        cv_.wait(lock, [this] { return stop_ || !tasks_.empty(); });
        if (stop_ && tasks_.empty()) return;
        task = std::move(tasks_.front());
        tasks_.pop();
      }
      task();
    }
  }

  std::vector<std::thread> workers_;
  std::queue<std::function<void()>> tasks_;
  mutable std::mutex mutex_;
  std::condition_variable cv_;
  std::atomic<bool> stop_{false};
};

}  // namespace ai_gateway
