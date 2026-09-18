#!/usr/bin/env bash
# 下载语义缓存评测用的公开数据集到 scripts/datasets/（**不入 git**，见 .gitignore）
#
# 数据来源与许可
# ---------------------------------------------------------------------------
# 1. LCQMC —— 中文同义句判别数据集（Large-scale Chinese Question Matching Corpus）
#     论文：Xin Liu et al., "LCQMC: A Large-scale Chinese Question Matching Corpus",
#           COLING 2018.  https://aclanthology.org/C18-1166/
#     本脚本取 C-MTEB 的整理版本（Maven/MTEB 的中文检索评测集之一）：
#       https://huggingface.co/datasets/C-MTEB/LCQMC
#     字段：sentence1, sentence2, score（1 = 同义）
#     许可：Apache-2.0（随 C-MTEB 仓库声明；原始 LCQMC 由哈工大智能技术与自然
#           语言处理实验室发布，仅限研究用途，商用请自行确认）
# 2. PAWS-X 中文 —— 对抗式改写句对（Paraphrase Adversaries from Word Scrambling）
#     论文：Yinfei Yang et al., "PAWS-X: A Cross-lingual Adversarial Dataset for
#           Paraphrase Identification", EMNLP 2019.  https://arxiv.org/abs/1908.11828
#       https://huggingface.co/datasets/google-research-datasets/paws-x
#     字段：id, sentence1, sentence2, label（1 = 同义）
#     许可：Apache-2.0（随 google-research-datasets/paws-x 仓库声明）
#
# 为什么走 hf-mirror.com：本机直连 huggingface.co 不可达（见项目记忆），
# hf-mirror.com 是 HF 的镜像，内容一致（本项目的 SHA256 校验值即取自镜像，
# 与官方 hub 取到的文件逐字节相同，已在 2026-09 复核）。
#
# 用法：
#   bash scripts/fetch_eval_datasets.sh            # 下载（已存在且校验通过则跳过）
#   bash scripts/fetch_eval_datasets.sh --force    # 忽略本地文件重新下载
#   bash scripts/fetch_eval_datasets.sh --verify   # 只校验
set -euo pipefail

# 用固定 commit 的 resolve URL：内容寻址，长期可复现（不用 main 这种会漂移的分支名）
HF_MIRROR="${HF_MIRROR:-https://hf-mirror.com}"
LCQMC_URL="${HF_MIRROR}/datasets/C-MTEB/LCQMC/resolve/17f9b096f80380fce5ed12a9be8be7784b337daf/data/validation-00000-of-00001-ae04bea7d65ea894.parquet"
PAWSX_URL="${HF_MIRROR}/datasets/google-research-datasets/paws-x/resolve/4cd8187c404bda33cb1f62b49b001115862acf37/zh/validation-00000-of-00001.parquet"

# 期望的 SHA256 / 字节数（下载后校验；不一致直接失败，不留下半个文件）
LCQMC_SHA="d2a4927bd2767ccd018dda33c650f86c074441337e7d73627325612a86379b4c"
LCQMC_BYTES=526804
PAWSX_SHA="9c830a0a82e411c3587537380f88abd53d7dba08624a1f3ec1da43bf8265fbef"
PAWSX_BYTES=346797

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEST_DIR="${SCRIPT_DIR}/datasets"
FORCE=0
VERIFY_ONLY=0
for a in "$@"; do
  case "$a" in
    --force) FORCE=1 ;;
    --verify) VERIFY_ONLY=1 ;;
    -h|--help) sed -n '2,40p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) echo "未知参数: $a（支持 --force / --verify）" >&2; exit 2 ;;
  esac
done

mkdir -p "$DEST_DIR"

sha_of() { sha256sum "$1" | awk '{print $1}'; }
size_of() { stat -c %s "$1"; }

# 校验一个文件：返回 0 = 存在且哈希/大小都对
check_file() {
  local path="$1" want_sha="$2" want_bytes="$3" name="$4"
  [[ -f "$path" ]] || { echo "缺失: ${name}（${path}）"; return 1; }
  local got_sha got_bytes
  got_sha="$(sha_of "$path")"
  got_bytes="$(size_of "$path")"
  if [[ "$got_sha" != "$want_sha" ]]; then
    echo "SHA256 不符: ${name}"
    echo "  期望 ${want_sha}"
    echo "  实际 ${got_sha}"
    return 1
  fi
  if [[ "$got_bytes" != "$want_bytes" ]]; then
    echo "字节数不符: ${name}（期望 ${want_bytes}，实际 ${got_bytes}）"
    return 1
  fi
  echo "OK: ${name}  sha256=${got_sha}  bytes=${got_bytes}"
  return 0
}

# 下载到临时文件 -> 校验 -> 原子改名。绝不把未校验的文件留在目标位置
fetch() {
  local url="$1" path="$2" want_sha="$3" want_bytes="$4" name="$5"
  local tmp="${path}.part"
  echo "下载 ${name}"
  echo "  ${url}"
  rm -f "$tmp"
  if command -v curl >/dev/null 2>&1; then
    curl -fL --retry 3 --retry-delay 2 --connect-timeout 15 -o "$tmp" "$url"
  elif command -v wget >/dev/null 2>&1; then
    wget -O "$tmp" "$url"
  else
    echo "[fatal] 需要 curl 或 wget" >&2
    exit 2
  fi
  local got_sha got_bytes
  got_sha="$(sha_of "$tmp")"
  got_bytes="$(size_of "$tmp")"
  if [[ "$got_sha" != "$want_sha" || "$got_bytes" != "$want_bytes" ]]; then
    echo "[fatal] 下载内容校验失败：${name}" >&2
    echo "        期望 sha256=${want_sha} bytes=${want_bytes}" >&2
    echo "        实际 sha256=${got_sha} bytes=${got_bytes}" >&2
    echo "        镜像可能未同步或已变更；已删除临时文件，请稍后重试或用" >&2
    echo "        HF_ENDPOINT=https://huggingface.co 直连（如可达）复核" >&2
    rm -f "$tmp"
    exit 1
  fi
  mv -f "$tmp" "$path"
  echo "  -> ${path}  sha256=${got_sha}  bytes=${got_bytes}"
}

lcqmc_path="${DEST_DIR}/lcqmc-val.parquet"
pawsx_path="${DEST_DIR}/pawsx-zh-val.parquet"

if [[ "$VERIFY_ONLY" == "1" ]]; then
  rc=0
  check_file "$lcqmc_path" "$LCQMC_SHA" "$LCQMC_BYTES" "LCQMC 验证集" || rc=1
  check_file "$pawsx_path" "$PAWSX_SHA" "$PAWSX_BYTES" "PAWS-X 中文验证集" || rc=1
  exit $rc
fi

if [[ "$FORCE" != "1" ]] && check_file "$lcqmc_path" "$LCQMC_SHA" "$LCQMC_BYTES" "LCQMC 验证集" >/dev/null 2>&1 \
   && check_file "$pawsx_path" "$PAWSX_SHA" "$PAWSX_BYTES" "PAWS-X 中文验证集" >/dev/null 2>&1; then
  echo "两数据集均已存在且校验通过，跳过下载（--force 可强制重新下载）"
  check_file "$lcqmc_path" "$LCQMC_SHA" "$LCQMC_BYTES" "LCQMC 验证集"
  check_file "$pawsx_path" "$PAWSX_SHA" "$PAWSX_BYTES" "PAWS-X 中文验证集"
else
  fetch "$LCQMC_URL" "$lcqmc_path" "$LCQMC_SHA" "$LCQMC_BYTES" "LCQMC 验证集"
  fetch "$PAWSX_URL" "$pawsx_path" "$PAWSX_SHA" "$PAWSX_BYTES" "PAWS-X 中文验证集"
fi

cat <<EOF

完成。下一步：
  python3 scripts/eval_semantic_cache.py --limit 3000

注意：这两个 .parquet 文件**不进 git**（.gitignore 已加规则 scripts/datasets/*.parquet），
      请在本机/CI 上用本脚本按需拉取。
EOF
