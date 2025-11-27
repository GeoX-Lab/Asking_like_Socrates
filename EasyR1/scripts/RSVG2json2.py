#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RSVG -> JSONL exporter (multi-choice version with shortcut expectation E)
- 读取 DIOR_RSVG 风格的 XML 和 splits（train/val/test）
- 针对每个 <object> 生成一题多选题：
  * problem: 固定中文题干 + 选项（A/B/C...，形如 "Q: ... | A: yes/no"）
  * answer: {
        "tag": "multi_choice",
        "bbox": null,
        "E": <设计随机性的期望奖励>,
        "labels": ["A","C",...],  # 正确选项
        "realized_r": <当前题真实奖励>,
        "k": <k参数>
    }
  * images: [img_abs]

- 选项由目标框几何属性自动生成（位置/大小/长宽比/是否包含短语等），
  再随机决定哪些填“正确答案”（保持至少一个正确与一个错误）。

依赖:
  pip install pillow tqdm datasets
"""

import argparse
import json
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import random
import math

from PIL import Image as PILImage
from tqdm import tqdm

# ------------------------- Utilities -------------------------

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def read_split_indices(splits_dir: Path, split: str) -> set:
    """Read DIOR_RSVG/{split}.txt (global object indices)."""
    split_file = splits_dir / f"{split}.txt"
    with open(split_file, "r", encoding="utf-8") as f:
        return {int(line.strip()) for line in f if line.strip()}

def list_xmls(anno_path: Path) -> List[Path]:
    """List all XML files (sorted)."""
    return sorted([p for p in anno_path.rglob("*.xml")])

def parse_bbox(member: ET.Element) -> Optional[Tuple[int, int, int, int]]:
    """Parse bbox from <object>, return (x1, y1, x2, y2)."""
    bnd = member.find("bndbox")
    if bnd is not None:
        try:
            xmin = int(float(bnd.find("xmin").text))
            ymin = int(float(bnd.find("ymin").text))
            xmax = int(float(bnd.find("xmax").text))
            ymax = int(float(bnd.find("ymax").text))
            return xmin, ymin, xmax, ymax
        except Exception:
            pass
    try:
        cand = member[2]
        vals = [int(float(cand[i].text)) for i in range(4)]
        return vals[0], vals[1], vals[2], vals[3]
    except Exception:
        return None

def parse_phrase(member: ET.Element) -> Optional[str]:
    """Parse phrase string from <object>."""
    for tag in ["content", "phrase", "text", "description", "name"]:
        node = member.find(tag)
        if node is not None and node.text:
            return node.text.strip()
    try:
        if len(member) >= 4 and member[3].text:
            return member[3].text.strip()
    except Exception:
        pass
    return None

def image_filename_from_root(root: ET.Element) -> Optional[str]:
    node = root.find("./filename")
    if node is not None and node.text:
        return node.text.strip()
    return None

def export_or_link_image(src_path: Path, dst_dir: Path, copy_images: bool) -> str:
    """Return absolute path string for training image."""
    if not copy_images:
        return src_path.resolve().as_posix()
    ensure_dir(dst_dir)
    dst_path = dst_dir / src_path.name
    if not dst_path.exists():
        shutil.copy2(src_path, dst_path)
    return dst_path.resolve().as_posix()

def get_image_wh(img_path: Path) -> Tuple[int, int]:
    with PILImage.open(img_path) as im:
        w, h = im.size
    if w <= 0 or h <= 0:
        raise ValueError(f"Invalid image size for {img_path}: {w}x{h}")
    return w, h

# ------------------------- Multi-choice tools -------------------------

def wants_for_len(L: int, rng: random.Random) -> List[bool]:
    """随机决定哪些选项填正确答案（True=此选项是正确项）。保证至少一个True和一个False。"""
    if L <= 1:
        return [True]
    w = [bool(rng.getrandbits(1)) for _ in range(L)]
    if all(w):            # 全 True，则翻一个为 False
        w[rng.randrange(L)] = False
    if not any(w):        # 全 False，则翻一个为 True
        w[rng.randrange(L)] = True
    return w

def expected_shortcut_reward_design(num_options: int, k: float) -> float:
    """
    只对“设计随机性”取期望：
    C ~ Binomial(m, 0.5),
    E[r] = sum_c Binom(m,c)/2^m * ((exp(k*c/m)-1)/(exp(k)-1))
    k=0 时退化为 0.5
    """
    m = int(num_options)
    if m <= 0:
        return 0.0
    if abs(k) < 1e-12:
        return 0.5
    denom = math.exp(k) - 1.0
    total = 0.0
    for c in range(m + 1):
        pc = math.comb(m, c) * (0.5 ** m)
        rc = (math.exp(k * (c / m)) - 1.0) / denom
        total += pc * rc
    return total

def realized_shortcut_reward(num_correct: int, num_options: int, k: float) -> float:
    """r = (exp(k*(c/m)) - 1) / (exp(k) - 1); k=0 时 r=c/m"""
    m = int(num_options)
    c = int(num_correct)
    if m <= 0:
        return 0.0
    if abs(k) < 1e-12:
        return c / m
    return (math.exp(k * (c / m)) - 1.0) / (math.exp(k) - 1.0)

def build_mc_options_from_geom(
    bx: List[float],
    phrase: str,
    rng: random.Random,
    min_opts: int = 5,
    max_opts: int = 12
) -> Tuple[List[str], List[int], List[str]]:
    """
    基于几何属性构造候选“判断题”（gold yes/no），
    再随机决定哪些用 gold，哪些用反转，从而得到 options & labels。
    返回:
      options_texts: ["Q: ... | A: yes", ...]
      labels_idx: [正确项的索引...]
      option_labels: ["A","B","C",...]
    """
    x1, y1, x2, y2 = bx
    cx = 0.5 * (x1 + x2)
    cy = 0.5 * (y1 + y2)
    w = max(1e-8, x2 - x1)
    h = max(1e-8, y2 - y1)
    area = w * h
    ar = w / h

    # 候选题（gold 答案）
    cands: List[Tuple[str, bool]] = []

    # 1) 存在该短语（一定为 True）
    cands.append((f"Does the image contain '{phrase}'?", True))
    # 2) 左半/右半
    cands.append(("Is the object's center in the left half (x<0.5)?", cx < 0.5))
    cands.append(("Is the object's center in the right half (x>=0.5)?", cx >= 0.5))
    # 3) 上半/下半
    cands.append(("Is the object's center in the top half (y<0.5)?", cy < 0.5))
    cands.append(("Is the object's center in the bottom half (y>=0.5)?", cy >= 0.5))
    # 4) 形状/面积阈值
    cands.append(("Is the object wider than tall (w>h)?", w > h))
    cands.append(("Is the object taller than wide (h>=w)?", h >= w))
    cands.append(("Is the bbox area >= 10% of the image?", area >= 0.10))
    cands.append(("Is the bbox area >= 20% of the image?", area >= 0.20))
    # 5) 纵横比
    cands.append(("Is the aspect ratio >= 1.5?", ar >= 1.5))
    cands.append(("Is the aspect ratio <= 0.67?", ar <= 0.67))

    # 去重（题干一致时保留第一次）
    seen = set()
    uniq = []
    for q, g in cands:
        if q not in seen:
            seen.add(q)
            uniq.append((q, g))
    cands = uniq

    # 选题数量
    L_all = len(cands)
    L = rng.randint(min_opts, max_opts)
    L = min(L, L_all)
    # 采样 L 道题
    sampled = rng.sample(cands, L)

    # wants: True -> 用 gold 答案（此选项为正确项）；False -> 反转（此选项为错误项）
    wants = wants_for_len(L, rng)

    options_texts: List[str] = []
    labels_idx: List[int] = []
    for i, (q, gold) in enumerate(sampled):
        ans = "yes" if (gold if wants[i] else (not gold)) else "no"
        options_texts.append(f"Q: {q} | A: {ans}")
        if wants[i]:
            labels_idx.append(i)

    # 至少一个正确+一个错误（保险）
    if not labels_idx or len(labels_idx) == L:
        # 强制第一项为正确
        labels_idx = [0]
        # 替换第一项答案为 gold
        q0, gold0 = sampled[0]
        options_texts[0] = f"Q: {q0} | A: {'yes' if gold0 else 'no'}"

    option_labels = [chr(65 + i) for i in range(L)]  # A,B,C,...
    return options_texts, labels_idx, option_labels

# ------------------------- Collect objects -------------------------

def iterate_objects(xml_paths: List[Path], images_root: Path) -> Tuple[int, List[Dict]]:
    """Iterate over XML <object>, return (total_objects, record list)."""
    global_count = 0
    records: List[Dict] = []
    for xml_path in tqdm(xml_paths, desc="Scanning XMLs"):
        try:
            root = ET.parse(xml_path).getroot()
        except Exception:
            continue

        fname = image_filename_from_root(root)
        if not fname:
            continue

        img_path = images_root / fname

        for member in root.findall("object"):
            bbox = parse_bbox(member)
            phrase = parse_phrase(member) or "object"
            records.append({
                "global_idx": global_count,
                "img_path": img_path,
                "bbox": bbox,
                "phrase": phrase,
            })
            global_count += 1

    return global_count, records

# ------------------------- Subsampling -------------------------

def sample_indices(indices: set, ratio: float, seed: int = 42, mode: str = "random") -> set:
    idx_list = sorted(indices)
    n = len(idx_list)
    if n == 0 or ratio >= 1.0:
        return set(idx_list)
    if ratio <= 0.0:
        raise ValueError("sample-ratio must be > 0.")
    k = max(1, int(n * ratio))
    if mode == "stride":
        step = max(1, round(n / k))
        sampled = idx_list[::step][:k]
    else:
        rng = random.Random(seed)
        sampled = rng.sample(idx_list, k)
    return set(sampled)

# ------------------------- Conversion -------------------------

def convert_split(
    records: List[Dict],
    indices: set,
    out_jsonl: Path,
    copy_images: bool,
    copy_dir: Path,
    decimals: int = 6,
    *,
    rng: Optional[random.Random] = None,
    mc_min: int = 5,
    mc_max: int = 12,
    k_param: float = 5.0,
) -> int:
    ensure_dir(out_jsonl.parent)
    if rng is None:
        rng = random.Random(0)

    kept = 0
    with open(out_jsonl, "w", encoding="utf-8") as f:
        for rec in records:
            if rec["global_idx"] not in indices:
                continue
            img_path: Path = rec["img_path"]
            if not img_path.exists():
                continue
            img_abs = export_or_link_image(img_path, copy_dir, copy_images)

            phrase = (rec["phrase"] or "object").strip()
            bbox = rec["bbox"]
            if bbox is None:
                continue

            # 读取原图尺寸并归一化 bbox
            try:
                w, h = get_image_wh(img_path)
            except Exception:
                continue

            x1, y1, x2, y2 = bbox
            if x2 < x1:
                x1, x2 = x2, x1
            if y2 < y1:
                y1, y2 = y2, y1
            x1 = max(0, min(x1, w))
            x2 = max(0, min(x2, w))
            y1 = max(0, min(y1, h))
            y2 = max(0, min(y2, h))
            if x2 <= x1 or y2 <= y1:
                continue

            bx = [x1/float(w), y1/float(h), x2/float(w), y2/float(h)]
            bx = [max(0.0, min(1.0, v)) for v in bx]
            if decimals is not None and decimals >= 0:
                bx = [round(v, decimals) for v in bx]

            # 基于几何属性生成多选题选项
            options_texts, labels_idx, option_letters = build_mc_options_from_geom(
                bx, phrase, rng, min_opts=mc_min, max_opts=mc_max
            )

            # 题干 + 带字母选项
            labeled_lines = [f"{lab}. {opt}" for lab, opt in zip(option_letters, options_texts)]
            problem_text = "针对上面的图像，下面选项中答案能够正确回答问题的有哪些。\n" + "\n".join(labeled_lines)

            # 奖励（k=0 处理线性退化）
            m = len(option_letters)
            c = len(labels_idx)
            E = expected_shortcut_reward_design(m, k_param)
            realized_r = realized_shortcut_reward(c, m, k_param)

            # 输出对象（按你的目标组织）
            obj = {
                "problem": problem_text,
                "answer": {
                    "tag": "multi_choice",
                    "E": E,                     # 设计随机性的期望奖励
                    "labels": [option_letters[i] for i in labels_idx],
                },
                "images": [img_abs],
            }
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
            kept += 1

    return kept

# ------------------------- CLI -------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Export RSVG to JSONL as multi-choice tasks with shortcut expectation."
    )
    parser.add_argument("--images", required=True, help="Root directory of images (matched with <filename> in XML)")
    parser.add_argument("--annos", required=True, help="Annotations root directory (with .xml files)")
    parser.add_argument("--splits", required=True, help="Directory containing train.txt / val.txt / test.txt (DIOR_RSVG style)")
    parser.add_argument("--out", required=True, help="Output directory")
    parser.add_argument("--copy-images", action="store_true",
                        help="Copy images to output directory (default: link original absolute path)")
    parser.add_argument("--decimals", type=int, default=6,
                        help="Number of decimals for normalized coordinates (default=6, negative to disable rounding)")
    parser.add_argument("--sample-ratio", type=float, default=0.5,
                        help="Subsampling ratio for each split (default=0.5)")
    parser.add_argument("--sample-seed", type=int, default=42,
                        help="Random seed for sampling (default=42)")
    parser.add_argument("--sample-mode", type=str, default="random", choices=["random", "stride"],
                        help="Sampling mode: random/stride (default random)")

    # 多选题控制
    parser.add_argument("--mc-min", type=int, default=5, help="最少选项数（默认5）")
    parser.add_argument("--mc-max", type=int, default=12, help="最多选项数（默认12）")

    # 奖励参数
    parser.add_argument("--k", type=float, default=5.0, help="指数化奖励参数 k（默认 5.0）")

    args = parser.parse_args()

    # clamp
    args.mc_min = max(2, int(args.mc_min))
    args.mc_max = max(args.mc_min, int(args.mc_max))
    if args.sample_ratio < 0 or args.sample_ratio > 1:
        raise ValueError("--sample-ratio must be in [0,1]")

    rng = random.Random(args.sample_seed)

    images_root = Path(args.images).expanduser().resolve()
    anno_root = Path(args.annos).expanduser().resolve()
    splits_dir = Path(args.splits).expanduser().resolve()
    out_dir = Path(args.out).expanduser().resolve()

    xmls = list_xmls(anno_root)
    total_objects, records = iterate_objects(xmls, images_root)
    print(f"[INFO] Parsed {len(xmls)} XMLs, total objects = {total_objects}")

    train_indices_full = read_split_indices(splits_dir, "train")
    val_indices_full   = read_split_indices(splits_dir, "val")
    test_indices_full  = read_split_indices(splits_dir, "test")

    train_indices = sample_indices(train_indices_full, args.sample_ratio, args.sample_seed, args.sample_mode)
    val_indices   = sample_indices(val_indices_full,   args.sample_ratio, args.sample_seed + 1, args.sample_mode)
    test_indices  = sample_indices(test_indices_full,  args.sample_ratio, args.sample_seed + 2, args.sample_mode)

    print(f"[INFO] Sampling train: {len(train_indices_full)} -> {len(train_indices)}")
    print(f"[INFO] Sampling val:   {len(val_indices_full)}   -> {len(val_indices)}")
    print(f"[INFO] Sampling test:  {len(test_indices_full)}  -> {len(test_indices)}")

    # -------- train --------
    train_jsonl = out_dir / "train.jsonl"
    train_img_dir = out_dir / "images" / "train"
    ensure_dir(train_img_dir)
    kept_train = convert_split(
        records, train_indices, train_jsonl, args.copy_images,
        train_img_dir, args.decimals,
        rng=rng, mc_min=args.mc_min, mc_max=args.mc_max, k_param=args.k,
    )
    print(f"[OK] train: wrote {kept_train} samples -> {train_jsonl}")

    # -------- val --------
    val_jsonl = out_dir / "val.jsonl"
    val_img_dir = out_dir / "images" / "val"
    ensure_dir(val_img_dir)
    kept_val = convert_split(
        records, val_indices, val_jsonl, args.copy_images,
        val_img_dir, args.decimals,
        rng=rng, mc_min=args.mc_min, mc_max=args.mc_max, k_param=args.k,
    )
    print(f"[OK] val:   wrote {kept_val} samples -> {val_jsonl}")

    # -------- test --------
    test_jsonl = out_dir / "test.jsonl"
    test_img_dir = out_dir / "images" / "test"
    ensure_dir(test_img_dir)
    kept_test = convert_split(
        records, test_indices, test_jsonl, args.copy_images,
        test_img_dir, args.decimals,
        rng=rng, mc_min=args.mc_min, mc_max=args.mc_max, k_param=args.k,
    )
    print(f"[OK] test:  wrote {kept_test} samples -> {test_jsonl}")

    print("\nDone. Example usage:")
    print(f"  data.train_files={out_dir.as_posix()}/train.jsonl \\")
    print(f"  data.val_files={out_dir.as_posix()}/val.jsonl \\")
    print(f"  data.test_files={out_dir.as_posix()}/test.jsonl \\")
    print("  data.image_dir=null")
    print("  data.prompt_key=problem")
    print("  data.answer_key=answer")
    print("  data.image_key=images")

if __name__ == "__main__":
    main()
