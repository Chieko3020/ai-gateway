// SSE usage 解析实现（设计约束见 include/server/sse_usage.h）
#include "server/sse_usage.h"

#include <cctype>

namespace ai_gateway {

namespace {

// 从 pos 开始跳过空白
size_t skip_ws(std::string_view s, size_t pos) {
  while (pos < s.size() &&
         (s[pos] == ' ' || s[pos] == '\t' || s[pos] == '\r' || s[pos] == '\n'))
    ++pos;
  return pos;
}

// 解析 pos 处的十进制整数。成功返回 true 并前移 pos（跳过数字）。
// 溢出检查省略：usage 语义上是个位数量级，任何能让 int64 溢出的输入都是
// 损坏/恶意数据，饱和成一个极大值不影响"是否统计到"的判定
bool parse_int(std::string_view s, size_t& pos, int64_t& out) {
  const size_t start = pos;
  bool neg = false;
  if (pos < s.size() && s[pos] == '-') {
    neg = true;
    ++pos;
  }
  int64_t v = 0;
  bool any = false;
  while (pos < s.size() && std::isdigit(static_cast<unsigned char>(s[pos]))) {
    v = v * 10 + (s[pos] - '0');
    ++pos;
    any = true;
  }
  if (!any) {
    pos = start;
    return false;
  }
  out = neg ? -v : v;
  return true;
}

struct UsageField {
  std::string_view key;
  int64_t* target;
};

}  // namespace

void SseUsageParser::feed(const char* data, size_t len) {
  if (len == 0) return;

  // 把上一轮留下的窗口拼到本块前面。窗口里可能有：
  //   - 一整行（还不完整，等本块补全）
  //   - 上一块的最后 kCarryBytes 字节（用于跨块的 key/value 配对）
  // carry_ 与 buf_ 是同一份数据的两段视图，这里统一成 buf_ 处理
  buf_.append(data, len);

  // 病态输入保护：一整块里连一个 '\n' 都没有且已经超过窗口上限，直接丢掉。
  // 正常 SSE 每条事件都带换行，这条路径只可能来自畸形/恶意上游
  if (buf_.find('\n') == std::string::npos) {
    if (buf_.size() > kMaxLineBytes) buf_.clear();
    return;
  }

  size_t start = 0;
  while (true) {
    const size_t nl = buf_.find('\n', start);
    if (nl == std::string::npos) break;  // 剩下的是未完行，留给下一次 feed
    std::string_view line(buf_.data() + start, nl - start);
    if (!line.empty() && line.back() == '\r') line.remove_suffix(1);
    process_line(line);
    start = nl + 1;
  }

  // 保留"未完行"继续等数据。同时把已完成部分的最后一小段也带上：
  // TCP 分块可能与事件边界无关，`"usage"` 与它的数字可能被切在两个 chunk 里
  // （此时后半段还没有 '\n'，整行会留在 buf_ 里，正常的跨块已被覆盖；
  //   这里额外保留 kCarryBytes 是为了覆盖"跨块的整行已被处理过"的边界情形）
  if (start > 0) {
    constexpr size_t kCarry = kCarryBytes;
    const size_t keep_from =
        (start > kCarry && start <= buf_.size()) ? start - kCarry : 0;
    buf_.erase(0, keep_from);
  }
  if (buf_.size() > kMaxLineBytes) buf_.clear();  // 病态长行：放弃
}

void SseUsageParser::process_line(std::string_view line) {
  // 只认 data: 行。SSE 还有 event:/id:/retry:/注释(:) 行，它们不含 usage
  constexpr std::string_view kData = "data:";
  if (line.size() < kData.size() || line.substr(0, kData.size()) != kData) return;

  std::string_view payload = line.substr(kData.size());
  if (payload.size() > kMaxLineBytes) return;

  // usage 只可能出现在 JSON 对象顶层的 "usage" 里。payload 里没有这个键就跳过：
  // 这一步挡掉"别的扩展字段恰好也叫 prompt_tokens"的情况
  if (payload.find("\"usage\"") == std::string_view::npos) return;

  const UsageField fields[] = {
      {"\"prompt_tokens\"", &usage_.prompt_tokens},
      {"\"completion_tokens\"", &usage_.completion_tokens},
  };
  for (const auto& f : fields) {
    const size_t kpos = payload.find(f.key);
    if (kpos == std::string_view::npos) continue;
    // 找第一个 ':' 后面的数字。中间不允许再出现 '{'（那说明是嵌套对象，
    // 不是 "key": 123 的形状），避免把嵌套结构里的数字误配到外层键上
    size_t pos = skip_ws(payload, kpos + f.key.size());
    if (pos >= payload.size() || payload[pos] != ':') continue;
    pos = skip_ws(payload, pos + 1);
    int64_t v = 0;
    if (!parse_int(payload, pos, v)) continue;
    *f.target = v;
    usage_.seen = true;
  }
}

}  // namespace ai_gateway
