#!/bin/bash
set -euo pipefail

# VRSBench -> JSONL 多个 k 导出

# ===== 路径配置 =====
ROOT_DIR="/g0001sr/lzy/VRSBench"
OUT_DIR_ROOT="/g0001sr/lzy/VRSBench_json_K"

# ===== 基础参数 =====
COPY_FLAG="--copy-images"          # 不想复制图片就设为空字符串 ""
PHRASE_SOURCE="ref"
SAMPLE_RATIO=1.0
SHORTCUT_MODE="exp"
SHORTCUT_SAMPLES=10000

# ===== 循环的 k 值 =====
KS=(0 2.5 5)

# ===== 创建根目录 =====
mkdir -p "$OUT_DIR_ROOT"

# ===== 工具函数：k转目录名 =====
k_tag () {
  local k_str="$1"
  echo "k_$(echo "$k_str" | sed 's/-/m/g; s/\./p/g')"
}

# ===== 循环执行 =====
for K in "${KS[@]}"; do
  TAG=$(k_tag "$K")
  OUT_DIR="${OUT_DIR_ROOT}/${TAG}"
  mkdir -p "$OUT_DIR"

  echo "==== 处理 k=${K} -> 输出到 ${OUT_DIR} ===="

  python3 /g0001sr/lzy/EasyR1/scripts/VRSBench2json.py \
    --root "$ROOT_DIR" \
    --out "$OUT_DIR" \
    $COPY_FLAG \
    --phrase-source "$PHRASE_SOURCE" \
    --sample-ratio "$SAMPLE_RATIO" \
    --shortcut-mode "$SHORTCUT_MODE" \
    --shortcut-samples "$SHORTCUT_SAMPLES" \
    --shortcut-exp-k "$K"

  echo "==== 完成 k=${K} ===="
done

echo "全部完成。输出目录：$OUT_DIR_ROOT"
