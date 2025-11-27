#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RSVG -> JSONL exporter (GROUNDING ONLY + Shortcut Expectation E)
- 读取 DIOR_RSVG/{split}.txt 过滤全局 object 索引
- 可子采样导出
- 解析每个 XML <object> 的 bbox，归一化并输出为 0..1000 的整数坐标
- problem 字段：仅提问目标短语的 bbox（值域 [0,1000]，整数）
- answer 字段：仅包含 {"bbox": [...], "E": float}
  * bbox: 0..1000 的整数（输出用）
  * E:    基于 [0,1] 浮点坐标计算（内部使用），按 --decimals 四舍五入
- images 字段：图片绝对路径（或拷贝后的路径）

依赖:
  pip install pillow tqdm
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
    """List all XML files (sorted for determinism)."""
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

# ------------------------- Shortcut Expectation (random boxes) -------------------------

def iou_norm(a: List[float], b: List[float]) -> float:
    """IoU for normalized boxes [x1,y1,x2,y2] in [0,1]."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_w = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    inter_h = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = inter_w * inter_h
    if inter <= 0.0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    if union <= 0.0:
        return 0.0
    return inter / union

def sample_random_boxes_norm(n: int, rng: random.Random, min_wh: float, max_wh: float) -> List[List[float]]:
    """
    Sample n random boxes in normalized coords.
    width,height ~ Uniform[min_wh, max_wh]; x1~U[0,1-w], y1~U[0,1-h].
    """
    boxes: List[List[float]] = []
    for _ in range(max(0, int(n))):
        w = rng.uniform(min_wh, max_wh)
        h = rng.uniform(min_wh, max_wh)
        w = min(max(w, 1e-6), 1.0)
        h = min(max(h, 1e-6), 1.0)
        x1 = rng.uniform(0.0, 1.0 - w)
        y1 = rng.uniform(0.0, 1.0 - h)
        x2 = x1 + w
        y2 = y1 + h
        boxes.append([x1, y1, x2, y2])
    return boxes

def shortcut_expectation_E(
    gt_box: List[float],
    n_samples: int,
    rng: random.Random,
    mode: str = "threshold",
    iou_thresh: float = 0.5,
    min_wh: float = 0.05,
    max_wh: float = 1.0,
    *,
    k: float = 5.0,  # exp 模式的拉伸参数
) -> float:
    """
    计算期望得分 E。
    - mode == "threshold": E = mean( 1{IoU >= iou_thresh} )
    - mode == "iou":       E = mean( IoU )
    - mode == "exp":       E = mean( (exp(k*IoU)-1) / (exp(k)-1) ), k→0 时退化为 mean(IoU)
    """
    if n_samples <= 0:
        return 0.0
    boxes = sample_random_boxes_norm(n_samples, rng, min_wh, max_wh)
    s = 0.0
    if mode == "iou":
        for rb in boxes:
            s += iou_norm(gt_box, rb)
    elif mode == "exp":
        # 数值稳定：k→0 时用极限 r≈IoU，避免 0/0
        if abs(k) < 1e-8:
            for rb in boxes:
                i = iou_norm(gt_box, rb)
                s += i
        else:
            denom = math.exp(k) - 1.0
            for rb in boxes:
                i = iou_norm(gt_box, rb)
                s += (math.exp(k * i) - 1.0) / denom
    else:
        thr = float(iou_thresh)
        for rb in boxes:
            s += 1.0 if iou_norm(gt_box, rb) >= thr else 0.0
    return s / float(n_samples)

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
            phrase = parse_phrase(member)
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

SCALE = 1000  # 坐标输出缩放到 0..1000 的整数

def _to_int_0_1000(x: float) -> int:
    """四舍五入并严格裁剪到 [0,1000] 区间。"""
    return max(0, min(SCALE, int(round(x))))

def convert_split(
    records: List[Dict],
    indices: set,
    out_jsonl: Path,
    copy_images: bool,
    copy_dir: Path,
    decimals: int = 6,   # 仅用于 E 的小数位；bbox 不再使用该参数
    *,
    # ---- only Shortcut Expectation controls kept ----
    shortcut_samples: int = 10000,
    shortcut_mode: str = "exp",
    shortcut_iou_thresh: float = 0.5,
    shortcut_min_wh: float = 0.05,
    shortcut_max_wh: float = 1.0,
    rng_shortcut: Optional[random.Random] = None,
    shortcut_exp_k: float = 5.0,  # 新增：exp 模式的 k
) -> int:
    ensure_dir(out_jsonl.parent)
    if rng_shortcut is None:
        rng_shortcut = random.Random(0)

    kept = 0
    with open(out_jsonl, "w", encoding="utf-8") as f:
        for rec in records:
            if rec["global_idx"] not in indices:
                continue
            img_path: Path = rec["img_path"]
            if not img_path.exists():
                continue
            img_abs = export_or_link_image(img_path, copy_dir, copy_images)
            try:
                w, h = get_image_wh(img_path)
            except Exception:
                continue

            phrase = (rec["phrase"] or "").strip()
            bbox = rec["bbox"]
            if bbox is None:
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

            # filter zero/negative area boxes
            if x2 <= x1 or y2 <= y1:
                continue

            # 1) 内部用浮点 [0,1] 计算 E
            bx_norm = [x1/float(w), y1/float(h), x2/float(w), y2/float(h)]
            bx_norm = [max(0.0, min(1.0, v)) for v in bx_norm]

            # 2) 输出用 0..1000 的整数
            bx_int = [_to_int_0_1000(v * SCALE) for v in bx_norm]

            # problem: grounding only (说明值域为 [0,1000] 且为整数)
            problem = (
                f"<image> What are the normalized bounding box coordinates "
                f"(x1, y1, x2, y2; integers in [0, {SCALE}]) for '{phrase}'?"
            ).strip()

            # --- Shortcut expectation E（基于 bx_norm 计算） ---
            E = shortcut_expectation_E(
                bx_norm,
                n_samples=shortcut_samples,
                rng=rng_shortcut,
                mode=shortcut_mode,
                iou_thresh=shortcut_iou_thresh,
                min_wh=shortcut_min_wh,
                max_wh=shortcut_max_wh,
                k=shortcut_exp_k,  # 传入 k
            )
            if decimals is not None and decimals >= 0:
                E = round(float(E), decimals)

            obj = {
                "problem": problem,
                "answer": {
                    "tag":"iou",
                    "bbox": bx_int,  # 0..1000 的整数
                    "E": E,
                },
                "images": [img_abs],
            }
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
            kept += 1
    return kept

# ------------------------- CLI -------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Export RSVG to JSONL with bbox as 0..1000 integers and Shortcut Expectation E (grounding only)."
    )
    parser.add_argument("--images", required=True, help="Root directory of images (matched with <filename> in XML)")
    parser.add_argument("--annos", required=True, help="Annotations root directory (with .xml files)")
    parser.add_argument("--splits", required=True, help="Directory containing train.txt / val.txt / test.txt (DIOR_RSVG style)")
    parser.add_argument("--out", required=True, help="Output directory")
    parser.add_argument("--copy-images", action="store_true",
                        help="Copy images to output directory (default: link original absolute path)")
    parser.add_argument("--decimals", type=int, default=6,
                        help="Number of decimals for E only (bbox are integers in [0,1000])")
    parser.add_argument("--sample-ratio", type=float, default=0.5,
                        help="Subsampling ratio for each split (default=0.5)")
    parser.add_argument("--sample-seed", type=int, default=42,
                        help="Random seed for sampling (default=42)")
    parser.add_argument("--sample-mode", type=str, default="random", choices=["random", "stride"],
                        help="Sampling mode: random/stride (default random)")

    # ---- Shortcut expectation controls ----
    parser.add_argument("--shortcut-samples", type=int, default=10000,
                        help="Number of random boxes per sample to estimate E (default=10000)")
    parser.add_argument("--shortcut-mode", type=str, default="exp", choices=["threshold", "iou", "exp"],
                        help="E scoring: 'threshold' uses 1{IoU>=t}, 'iou' uses IoU mean, 'exp' uses (exp(k*IoU)-1)/(exp(k)-1) (default='exp')")
    parser.add_argument("--shortcut-iou-thresh", type=float, default=0.5,
                        help="IoU threshold when --shortcut-mode=threshold (default=0.5)")
    parser.add_argument("--shortcut-min-wh", type=float, default=0.05,
                        help="Min width/height proportion for random boxes in [0,1] (default=0.05)")
    parser.add_argument("--shortcut-max-wh", type=float, default=1.0,
                        help="Max width/height proportion for random boxes in [0,1] (default=1.0)")
    parser.add_argument("--shortcut-seed", type=int, default=42,
                        help="Random seed for shortcut expectation")
    parser.add_argument("--shortcut-exp-k", type=float, default=5.0,
                        help="Scale factor k for exp mode: r=(exp(k*IoU)-1)/(exp(k)-1). When k→0, equals mean IoU.")

    args = parser.parse_args()

    # clamp shortcut params
    if args.shortcut_samples < 0:
        args.shortcut_samples = 0
    args.shortcut_min_wh = float(max(1e-6, min(1.0, args.shortcut_min_wh)))
    args.shortcut_max_wh = float(max(args.shortcut_min_wh, min(1.0, args.shortcut_max_wh)))
    if args.shortcut_mode == "threshold":
        args.shortcut_iou_thresh = float(max(0.0, min(1.0, args.shortcut_iou_thresh)))

    shortcut_rng = random.Random(args.shortcut_seed)

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
        shortcut_samples=args.shortcut_samples,
        shortcut_mode=args.shortcut_mode,
        shortcut_iou_thresh=args.shortcut_iou_thresh,
        shortcut_min_wh=args.shortcut_min_wh,
        shortcut_max_wh=args.shortcut_max_wh,
        rng_shortcut=shortcut_rng,
        shortcut_exp_k=args.shortcut_exp_k,
    )
    print(f"[OK] train: wrote {kept_train} samples -> {train_jsonl}")

    # -------- val --------
    val_jsonl = out_dir / "val.jsonl"
    val_img_dir = out_dir / "images" / "val"
    ensure_dir(val_img_dir)
    kept_val = convert_split(
        records, val_indices, val_jsonl, args.copy_images,
        val_img_dir, args.decimals,
        shortcut_samples=args.shortcut_samples,
        shortcut_mode=args.shortcut_mode,
        shortcut_iou_thresh=args.shortcut_iou_thresh,
        shortcut_min_wh=args.shortcut_min_wh,
        shortcut_max_wh=args.shortcut_max_wh,
        rng_shortcut=shortcut_rng,
        shortcut_exp_k=args.shortcut_exp_k,
    )
    print(f"[OK] val:   wrote {kept_val} samples -> {val_jsonl}")

    # -------- test --------
    test_jsonl = out_dir / "test.jsonl"
    test_img_dir = out_dir / "images" / "test"
    ensure_dir(test_img_dir)
    kept_test = convert_split(
        records, test_indices, test_jsonl, args.copy_images,
        test_img_dir, args.decimals,
        shortcut_samples=args.shortcut_samples,
        shortcut_mode=args.shortcut_mode,
        shortcut_iou_thresh=args.shortcut_iou_thresh,
        shortcut_min_wh=args.shortcut_min_wh,
        shortcut_max_wh=args.shortcut_max_wh,
        rng_shortcut=shortcut_rng,
        shortcut_exp_k=args.shortcut_exp_k,
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
