#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VRSBench -> JSONL exporter (GROUNDING ONLY + Shortcut Expectation E with controllable k)

目录结构（示例）:
  VRSBench/
    Annotations_train/*.json
    Annotations_val/*.json
    Images_train/*.png
    Images_val/*.png

保留：
- 子采样（--sample-ratio / --sample-mode）
- 可选 DIOR 风格 split 过滤（--splits 下 train/val/test.txt；索引以当前 split 为基准）
- 归一化坐标、图片复制/链接
- 捷径期望 E（Shortcut Expectation）：threshold / iou / exp
  - exp 形状可通过 --shortcut-exp-k 控制： (exp(k*IoU)-1)/(exp(k)-1)

输出 JSONL 每行：
{
  "problem": "<image> What are the normalized bounding box coordinates ... for '<phrase>'?",
  "answer": {"bbox": [x1,y1,x2,y2], "E": float},
  "images": ["/abs/path/to/image.png"]
}

依赖:
  pip install pillow tqdm
"""

import argparse
import json
import math
import random
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from PIL import Image as PILImage
from tqdm import tqdm

# ------------------------- Utilities -------------------------

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def read_split_indices(splits_dir: Optional[Path], split: str) -> Optional[set]:
    """读取 DIOR 风格 {split}.txt（全局 object 索引）。splits_dir 为空或文件缺失则返回 None。"""
    if not splits_dir:
        return None
    split_file = splits_dir / f"{split}.txt"
    if not split_file.exists():
        return None
    with open(split_file, "r", encoding="utf-8") as f:
        return {int(line.strip()) for line in f if line.strip()}

def export_or_link_image(src_path: Path, dst_dir: Path, copy_images: bool) -> str:
    """返回图片绝对路径；若 copy_images=True 则复制并返回复制后的绝对路径。"""
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

def clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))

def norm_box_order(x1: float, y1: float, x2: float, y2: float) -> Tuple[float,float,float,float]:
    if x2 < x1: x1, x2 = x2, x1
    if y2 < y1: y1, y2 = y2, y1
    return x1, y1, x2, y2

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
    """采样 n 个随机框（归一化）。w,h~U[min_wh,max_wh]；x1~U[0,1-w], y1~U[0,1-h]。"""
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
    exp_k: float = 5.0,  # 新增：exp 模式形状参数 k
) -> float:
    if n_samples <= 0:
        return 0.0
    boxes = sample_random_boxes_norm(n_samples, rng, min_wh, max_wh)
    s = 0.0
    if mode == "iou":
        for rb in boxes:
            s += iou_norm(gt_box, rb)
    elif mode == "exp":
        k = float(exp_k)
        if abs(k) < 1e-8:
            # k->0 极限：回到平均 IoU
            for rb in boxes:
                s += iou_norm(gt_box, rb)
        else:
            denom = math.expm1(k)  # exp(k) - 1
            for rb in boxes:
                i = iou_norm(gt_box, rb)
                s += math.expm1(k * i) / denom  # (exp(k*i)-1)/(exp(k)-1)
    else:  # "threshold"
        thr = float(iou_thresh)
        for rb in boxes:
            s += 1.0 if iou_norm(gt_box, rb) >= thr else 0.0
    return s / float(n_samples)

# ------------------------- Collect objects from VRSBench JSON -------------------------

def list_ann_jsons(root: Path, split: str) -> List[Path]:
    """
    同时支持：
      1) Annotations_{split}/*.json  多文件
      2) Annotations_{split}.json    单文件（内容是 list）
    """
    multi_dir = root / f"Annotations_{split}"
    single_file = root / f"Annotations_{split}.json"
    paths: List[Path] = []
    if multi_dir.exists():
        paths.extend(sorted(multi_dir.glob("*.json")))
    if single_file.exists():
        paths.append(single_file)
    return paths

def _yield_entries_from_json(path: Path):
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return
    if isinstance(data, list):
        for x in data:
            if isinstance(x, dict):
                yield x
    elif isinstance(data, dict):
        yield data

# ------------------------- Scale helpers ([0,1]/[0,100]/[0,1000]) -------------------------

def _detect_scale(vals):
    eps = 1e-9
    vmin, vmax = min(vals), max(vals)
    if vmin >= -eps and vmax <= 1.0 + eps:
        return 1000.0
    if vmin >= -eps and vmax <= 100.0 + eps:
        return 10.0
    return 1.0

def _clip1000i(x: float) -> int:
    return max(0, min(1000, int(round(x))))

def _order_xyxyi(x1: int, y1: int, x2: int, y2: int):
    if x2 < x1: x1, x2 = x2, x1
    if y2 < y1: y1, y2 = y2, y1
    return x1, y1, x2, y2

def _parse_ground_truth_box_0_1000(s: Optional[str]) -> Optional[List[int]]:
    """解析 RSBench 的 ground_truth: '{<25><40><33><60>}' -> [x1,y1,x2,y2] in [0,1000] (int)"""
    if not s or not isinstance(s, str):
        return None
    import re
    nums = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", s)
    if len(nums) < 4:
        return None
    x1, y1, x2, y2 = map(float, nums[:4])
    scale = _detect_scale([x1,y1,x2,y2])
    x1, y1, x2, y2 = (x1*scale, y1*scale, x2*scale, y2*scale)
    xi1, yi1, xi2, yi2 = map(_clip1000i, (x1, y1, x2, y2))
    xi1, yi1, xi2, yi2 = _order_xyxyi(xi1, yi1, xi2, yi2)
    if xi2 <= xi1 or yi2 <= yi1:
        return None
    return [xi1, yi1, xi2, yi2]

def _polygon_to_bbox_0_1000(coords: List[float]) -> Optional[List[int]]:
    """
    obj_corner: [x1,y1,x2,y2,x3,y3,x4,y4]，可能为 [0,1]/[0,100]/[0,1000]
    取外接矩形，转为 int [0,1000]
    """
    if not (isinstance(coords, list) and len(coords) >= 8):
        return None
    xs = coords[0::2][:4]
    ys = coords[1::2][:4]
    scale = _detect_scale(xs + ys)
    xs = [x*scale for x in xs]
    ys = [y*scale for y in ys]
    xi1, yi1 = _clip1000i(min(xs)), _clip1000i(min(ys))
    xi2, yi2 = _clip1000i(max(xs)), _clip1000i(max(ys))
    xi1, yi1, xi2, yi2 = _order_xyxyi(xi1, yi1, xi2, yi2)
    if xi2 <= xi1 or yi2 <= yi1:
        return None
    return [xi1, yi1, xi2, yi2]

# ------------------------- RSBench iterator -------------------------

def iterate_objects_vrsbench(
    root: Path,
    split: str,
    phrase_source: str = "auto"  # "ref" | "cls" | "auto"
) -> Tuple[int, List[Dict]]:
    """
    VRSBench：一图多目标。
    - 顶层: {"image": "...", "objects": [{...}]}
    - 每个 object 产一条记录
    - 框优先 obj_coord(0~1) -> *1000 转 int；否则 obj_corner(0~1 多边形) 取外接框
    - phrase 按 phrase_source 选择
    - 输出 bbox: int [0,1000]
    """
    ann_paths = list_ann_jsons(root, split)
    global_count = 0
    records: List[Dict] = []

    miss_img = miss_box = bad_json = 0

    for ap in tqdm(ann_paths, desc=f"Scanning {split} JSONs"):
        for item in _yield_entries_from_json(ap):
            if not isinstance(item, dict) or not isinstance(item.get("objects"), list):
                bad_json += 1
                print("bad_json")
                continue

            img_name = (item.get("image") or "").strip()
            if not img_name:
                bad_json += 1
                print("bad_json")
                continue

            img_path = image_path_for(root, split, img_name)
            if not img_path.exists():
                miss_img += 1
                print("miss_img")
                continue
            try:
                _ = get_image_wh(img_path)
            except Exception:
                miss_img += 1
                print("miss_img")
                continue

            for obj in (item.get("objects") or []):
                # 选择文本
                if phrase_source == "ref":
                    phrase = (obj.get("referring_sentence") or "").strip()
                elif phrase_source == "cls":
                    phrase = (obj.get("obj_cls") or "").strip()
                else:
                    phrase = (obj.get("referring_sentence") or obj.get("obj_cls") or "").strip()
                phrase = phrase.lower()

                # 取框：优先 obj_coord (0~1)，否则 obj_corner (0~1 多边形)
                box = None
                coord = obj.get("obj_coord")
                if isinstance(coord, list) and len(coord) == 4:
                    x1, y1, x2, y2 = map(float, coord)
                    xi1, yi1, xi2, yi2 = map(_clip1000i, (x1 * 1000.0, y1 * 1000.0, x2 * 1000.0, y2 * 1000.0))
                    xi1, yi1, xi2, yi2 = _order_xyxyi(xi1, yi1, xi2, yi2)
                    if xi2 > xi1 and yi2 > yi1:
                        box = [xi1, yi1, xi2, yi2]

                if box is None and isinstance(obj.get("obj_corner"), list):
                    box = _polygon_to_bbox_0_1000(obj["obj_corner"])

                if box is None:
                    miss_box += 1
                    print("miss_box")
                    continue

                records.append({
                    "global_idx": global_count,
                    "img_path": img_path,
                    "bbox": box,   # int [0,1000]
                    "phrase": phrase,
                })
                global_count += 1

    print(f"[DEBUG] {split}: records={len(records)}, miss_img={miss_img}, miss_box={miss_box}, bad_json={bad_json}")
    return global_count, records

def image_path_for(root: Path, split: str, filename: str) -> Path:
    """根据 split 找到对应图片路径（优先 Images_{split}，备选 Images/）"""
    p = root / f"Images_{split}" / filename
    if p.exists():
        return p
    alt = root / "Images" / filename
    return alt
# ------------------------- Subsampling -------------------------

def sample_indices(indices: List[int], ratio: float, seed: int = 42, mode: str = "random") -> set:
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
    indices: Optional[set],          # None 表示不过滤
    out_jsonl: Path,
    copy_images: bool,
    copy_dir: Path,
    decimals: int = 6,
    *,
    shortcut_samples: int = 10000,
    shortcut_mode: str = "exp",
    shortcut_iou_thresh: float = 0.5,
    shortcut_min_wh: float = 0.05,
    shortcut_max_wh: float = 1.0,
    rng_shortcut: Optional[random.Random] = None,
    shortcut_exp_k: float = 5.0,
) -> int:
    ensure_dir(out_jsonl.parent)
    if rng_shortcut is None:
        rng_shortcut = random.Random(0)

    kept = 0
    first_logged = False  # <<< 新增：只打印首条

    with open(out_jsonl, "w", encoding="utf-8") as f:
        for rec in records:
            if indices is not None and rec["global_idx"] not in indices:
                continue

            img_path: Path = rec["img_path"]
            if not img_path.exists():
                continue
            img_abs = export_or_link_image(img_path, copy_dir, copy_images)

            phrase = rec["phrase"] or ""

            # 已是 [0,1000] 整数，直接使用
            bx_1000 = list(map(int, rec["bbox"]))
            x1, y1, x2, y2 = bx_1000
            if x2 < x1: x1, x2 = x2, x1
            if y2 < y1: y1, y2 = y2, y1
            x1 = max(0, min(1000, x1))
            y1 = max(0, min(1000, y1))
            x2 = max(0, min(1000, x2))
            y2 = max(0, min(1000, y2))
            bx_1000 = [x1, y1, x2, y2]

            # E 计算用 0~1
            bx01 = [v / 1000.0 for v in bx_1000]
            E = shortcut_expectation_E(
                bx01,
                n_samples=shortcut_samples,
                rng=rng_shortcut,
                mode=shortcut_mode,
                iou_thresh=shortcut_iou_thresh,
                min_wh=shortcut_min_wh,
                max_wh=shortcut_max_wh,
                exp_k=shortcut_exp_k,
            )
            if decimals is not None and decimals >= 0:
                E = round(float(E), decimals)

            obj = {
                "problem": (
                    f"<image> What are the normalized bounding box coordinates "
                    f"(x1, y1, x2, y2; values in [0,1000]) for '{phrase}'?"
                ).strip(),
                "answer": {"tag": "iou", "bbox": bx_1000, "E": E},
                "images": [img_abs],
            }

            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
            kept += 1

            # <<< 新增：写完第一条后打印一条预览日志（只打印一次）
            if not first_logged:
                print("[FIRST SAMPLE PREVIEW]", json.dumps(obj, ensure_ascii=False), flush=True)
                first_logged = True

    return kept
# ------------------------- CLI -------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Export VRSBench to JSONL with normalized boxes and Shortcut Expectation E (grounding only, controllable exp-k)."
    )
    parser.add_argument("--root", required=True, help="VRSBench 根目录，包含 Annotations_{train,val} 与 Images_{train,val}")
    parser.add_argument("--out", required=True, help="输出目录")
    parser.add_argument("--copy-images", action="store_true", help="复制图片到输出目录（默认仅写绝对原路径）")
    parser.add_argument("--decimals", type=int, default=6, help="归一化坐标与 E 的小数位数（<0 关闭四舍五入）")
    parser.add_argument("--sample-ratio", type=float, default=1.0, help="子采样比例 (0,1]；默认全量")
    parser.add_argument("--sample-seed", type=int, default=42, help="采样随机种子")
    parser.add_argument("--sample-mode", type=str, default="random", choices=["random", "stride"], help="子采样方式")
    parser.add_argument("--phrase-source", type=str, default="auto", choices=["auto","ref","cls"], help="短语来源：referring_sentence / obj_cls / auto优先ref")
    parser.add_argument("--splits", type=str, default=None, help="可选：DIOR 风格 splits 目录（包含 train.txt/val.txt/test.txt）")

    # Shortcut E controls
    parser.add_argument("--shortcut-samples", type=int, default=10000)
    parser.add_argument("--shortcut-mode", type=str, default="exp", choices=["threshold", "iou", "exp"])
    parser.add_argument("--shortcut-iou-thresh", type=float, default=0.5)
    parser.add_argument("--shortcut-min-wh", type=float, default=0.05)
    parser.add_argument("--shortcut-max-wh", type=float, default=1.0)
    parser.add_argument("--shortcut-seed", type=int, default=42)
    parser.add_argument("--shortcut-exp-k", type=float, default=5.0,
                        help="k in exp shaping: (exp(k*IoU)-1)/(exp(k)-1). k≈0 退化为平均 IoU")

    args = parser.parse_args()

    # clamp shortcut params
    if args.shortcut_samples < 0:
        args.shortcut_samples = 0
    args.shortcut_min_wh = float(max(1e-6, min(1.0, args.shortcut_min_wh)))
    args.shortcut_max_wh = float(max(args.shortcut_min_wh, min(1.0, args.shortcut_max_wh)))
    if args.shortcut_mode == "threshold":
        args.shortcut_iou_thresh = float(max(0.0, min(1.0, args.shortcut_iou_thresh)))
    # 可选：限制 k 为非负
    args.shortcut_exp_k = max(0.0, float(args.shortcut_exp_k))

    shortcut_rng = random.Random(args.shortcut_seed)

    root = Path(args.root).expanduser().resolve()
    out_dir = Path(args.out).expanduser().resolve()
    ensure_dir(out_dir)

    for split in ["train", "val"]:
        total_objects, records = iterate_objects_vrsbench(root, split, phrase_source=args.phrase_source)
        print(f"[INFO] {split}: parsed objects = {total_objects}")

        # 可选 DIOR 风格 split 过滤
        splits_dir = Path(args.splits).expanduser().resolve() if args.splits else None
        indices_full = read_split_indices(splits_dir, split) if splits_dir else None

        # 子采样集合
        if indices_full is not None:
            base_indices = sorted(list(indices_full))
        else:
            base_indices = [rec["global_idx"] for rec in records]

        sampled = sample_indices(
            base_indices,
            args.sample_ratio,
            args.sample_seed + (0 if split == "train" else 1),
            args.sample_mode,
        )

        # 输出
        split_jsonl = out_dir / f"{split}.jsonl"
        split_img_dir = out_dir / "images" / split
        ensure_dir(split_img_dir)
        kept = convert_split(
            records, sampled if sampled else None,
            split_jsonl, args.copy_images, split_img_dir, args.decimals,
            shortcut_samples=args.shortcut_samples,
            shortcut_mode=args.shortcut_mode,
            shortcut_iou_thresh=args.shortcut_iou_thresh,
            shortcut_min_wh=args.shortcut_min_wh,
            shortcut_max_wh=args.shortcut_max_wh,
            rng_shortcut=shortcut_rng,
            shortcut_exp_k=args.shortcut_exp_k,  # 传入 k
        )
        print(f"[OK] {split}: wrote {kept} samples -> {split_jsonl}")

    print("\nDone. Example usage:")
    print(f"  data.train_files={out_dir.as_posix()}/train.jsonl \\")
    print(f"  data.val_files={out_dir.as_posix()}/val.jsonl \\")
    print("  data.image_dir=null")
    print("  data.prompt_key=problem")
    print("  data.answer_key=answer")
    print("  data.image_key=images")

if __name__ == "__main__":
    main()
