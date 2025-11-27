#!/bin/bash
set -euo pipefail

# RSVG -> JSONL 多个 k 循环导出

# ===== 路径配置 =====
IMAGES_DIR="/g0001sr/lzy/DIOR_RSVG/JPEGImages"
ANNOS_DIR="/g0001sr/lzy/DIOR_RSVG/Annotations"
SPLITS_DIR="/g0001sr/lzy/DIOR_RSVG"       # 目录下需包含 train.txt / val.txt / test.txt
OUT_DIR_ROOT="/g0001sr/lzy/DIOR_RSVG_jsonl_NEW_shortcut_K"   # 根目录；每个 k 会放到子目录下

# 是否复制图片到输出目录（加上 --copy-images 就会把图片拷贝到 OUT_DIR/images/{split}/）
COPY_FLAG="--copy-images"   # 不想复制就把这一行改为：COPY_FLAG=""

# 评价模式（threshold / iou / exp）
SHORTCUT_MODE="exp"

# 随机种子
SAMPLE_SEED=42
CONTROL_SEED=42     # Python 不使用此参数，留作记录
SHORTCUT_SEED=42

# ===== 循环的 k 值列表 =====
# 可按需修改：示例为 0.5、1、2.5、5、10
KS=(0 2.5 5)

# ===== 创建根输出目录 =====
mkdir -p "$OUT_DIR_ROOT"

# ===== 工具函数：把 2.5 变成 k_2p5，防止目录名带小数点 =====
k_tag () {
  local k_str="$1"
  # 把 '.' 替换为 'p'，把 '-' 替换为 'm'（如果你要用负数 k）
  echo "k_$(echo "$k_str" | sed 's/-/m/g; s/\./p/g')"
}

# ===== 执行循环 =====
for K in "${KS[@]}"; do
  TAG=$(k_tag "$K")
  OUT_DIR="${OUT_DIR_ROOT}/${TAG}"
  mkdir -p "$OUT_DIR"

  echo "==== 导出 k=${K} -> ${OUT_DIR} ===="

  python3 /g0001sr/lzy/EasyR1/scripts/RSVG2json.py \
    --images "$IMAGES_DIR" \
    --annos "$ANNOS_DIR" \
    --splits "$SPLITS_DIR" \
    --out "$OUT_DIR" \
    --sample-ratio 1 \
    --sample-seed "$SAMPLE_SEED" \
    --shortcut-seed "$SHORTCUT_SEED" \
    --shortcut-mode "$SHORTCUT_MODE" \
    --shortcut-exp-k "$K" \
    $COPY_FLAG

  echo "==== 完成 k=${K} ===="
done

echo "所有 k 导出完成。根目录：$OUT_DIR_ROOT"
