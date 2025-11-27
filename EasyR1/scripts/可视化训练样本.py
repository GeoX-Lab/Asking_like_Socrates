#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
从 JSONL 中随机抽取 k 张图片，将 0~1000 归一化 bbox 映射到像素坐标后可视化并保存。

用法示例：
python viz_norm1000_bbox_sample_k.py \
  --jsonl /g0001sr/lzy/datasets/all_for_train_k/k_0/val.jsonl \
  --outdir /g0001sr/lzy/datasets/all_for_train_k/k_0/vis_val_k20 \
  --k 20 --seed 42
"""

import os
import re
import json
import argparse
import random
from typing import List, Tuple, Optional

from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def norm1000_to_px(bbox_norm1000: List[float], w: int, h: int) -> Tuple[int, int, int, int]:
    if len(bbox_norm1000) != 4:
        raise ValueError(f"bbox 长度应为 4，收到：{bbox_norm1000}")
    x1n, y1n, x2n, y2n = bbox_norm1000
    x1n = clamp(x1n, 0, 1000)
    y1n = clamp(y1n, 0, 1000)
    x2n = clamp(x2n, 0, 1000)
    y2n = clamp(y2n, 0, 1000)

    x1 = int(round(x1n / 1000.0 * w))
    y1 = int(round(y1n / 1000.0 * h))
    x2 = int(round(x2n / 1000.0 * w))
    y2 = int(round(y2n / 1000.0 * h))

    x1, x2 = sorted([clamp(x1, 0, w), clamp(x2, 0, w)])
    y1, y2 = sorted([clamp(y1, 0, h), clamp(y2, 0, h)])
    return int(x1), int(y1), int(x2), int(y2)


def draw_bbox(
    img: Image.Image,
    bbox_xyxy: Tuple[int, int, int, int],
    label: Optional[str] = None,
    color: tuple = (255, 0, 0),
    width: int = 4,
) -> Image.Image:
    draw = ImageDraw.Draw(img)
    draw.rectangle(bbox_xyxy, outline=color, width=width)

    if label:
        try:
            font = ImageFont.truetype("DejaVuSans.ttf", size=max(12, int(min(img.size) * 0.02)))
        except Exception:
            font = ImageFont.load_default()

        text_w, text_h = draw.textbbox((0, 0), label, font=font)[2:]
        x1, y1, _, _ = bbox_xyxy
        pad = 2
        box_xyxy = (x1, max(0, y1 - text_h - 2 * pad), x1 + text_w + 2 * pad, y1)
        draw.rectangle(box_xyxy, fill=color)
        draw.text((x1 + pad, box_xyxy[1] + pad), label, fill=(255, 255, 255), font=font)
    return img


def extract_phrase_from_problem(problem: str) -> Optional[str]:
    m = re.search(r"[\"']([^\"']+)[\"']", problem)
    if m:
        return m.group(1).strip()
    return None


def process_jsonl_sample_k(jsonl_path: str, outdir: str, k: int, seed: Optional[int],
                           thickness: int = 4, suffix: str = "_vis") -> None:
    ensure_dir(outdir)

    # 读取所有行并打乱
    with open(jsonl_path, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f if ln.strip()]

    if seed is not None:
        random.seed(seed)
    random.shuffle(lines)

    success = 0
    tried = 0
    total_available = len(lines)

    pbar = tqdm(lines, desc=f"Sampling & visualizing (target k={k})")
    for idx, line in enumerate(pbar):
        tried += 1
        if success >= k:
            break

        try:
            obj = json.loads(line)
        except Exception as e:
            continue

        images = obj.get("images") or []
        if not images:
            continue

        img_path = images[0]
        if not os.path.isfile(img_path):
            continue

        answer = obj.get("answer") or {}
        bbox = answer.get("bbox")
        if not bbox or len(bbox) != 4:
            continue

        tag = answer.get("tag", "")
        E = answer.get("E", None)
        problem = obj.get("problem", "")

        try:
            with Image.open(img_path) as im:
                im = im.convert("RGB")
                w, h = im.size
                x1, y1, x2, y2 = norm1000_to_px(bbox, w, h)

                phrase = extract_phrase_from_problem(problem)
                parts = []
                if tag:
                    parts.append(f"tag:{tag}")
                if E is not None:
                    parts.append(f"E:{E:.6f}")
                if phrase:
                    parts.append(phrase)
                label = " | ".join(parts) if parts else None

                out_img = draw_bbox(im, (x1, y1, x2, y2), label=label, width=thickness)

                base = os.path.basename(img_path)
                stem, ext = os.path.splitext(base)
                out_path = os.path.join(outdir, f"{stem}{suffix}{ext if ext else '.jpg'}")
                out_img.save(out_path, quality=95)

                success += 1
                pbar.set_postfix_str(f"ok={success}/{k}")
        except Exception:
            continue

    print(f"\n目标 k = {k}；成功可视化 {success} 张；尝试 {tried} 行 / 总计 {total_available} 行。输出目录：{outdir}")


def main():
    parser = argparse.ArgumentParser(description="随机抽取 k 张并可视化 0~1000 归一化 bbox")
    parser.add_argument("--jsonl", required=True, help="输入 JSONL 路径，例如 /g0001sr/lzy/datasets/all_for_train_k/k_0/val.jsonl")
    parser.add_argument("--outdir", required=True, help="输出图片目录")
    parser.add_argument("--k", type=int, required=True, help="随机抽取并成功可视化的目标张数")
    parser.add_argument("--seed", type=int, default=None, help="随机种子（可选，设定后可复现）")
    parser.add_argument("--thickness", type=int, default=4, help="框线宽(像素)")
    parser.add_argument("--suffix", type=str, default="_vis", help="输出文件名后缀")
    args = parser.parse_args()

    if args.k <= 0:
        raise ValueError("参数 --k 必须为正整数。")

    process_jsonl_sample_k(
        jsonl_path=args.jsonl,
        outdir=args.outdir,
        k=args.k,
        seed=args.seed,
        thickness=args.thickness,
        suffix=args.suffix,
    )


if __name__ == "__main__":
    main()
"""
python /g0001sr/lzy/EasyR1/scripts/可视化训练样本.py \
  --jsonl /g0001sr/lzy/VRSBench_json_K/k_0/val.jsonl\
  --outdir /g0001sr/lzy/EasyR1/scripts/可视化结果 \
  --k 100 --seed 42
"""