#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Merge multiple grounding datasets with JSONL + images layout.

Each source dataset is expected to look like:
  <src_dir>/
    train.jsonl
    val.jsonl
    test.jsonl
    images/
      train/...
      val/...
      test/...

What it does:
- For each split (train/val/test), read JSONL lines from all sources
- For every record, copy (or symlink) each image into:
      <dst>/images/<split>/<prefix>_<basename>
  where <prefix> defaults to the source folder name
- Rewrites the JSON's "images" field to the new absolute path(s)
- Appends the (unchanged) metadata including your "answer" dict (bbox, E, etc.)
- Writes merged JSONLs to:
      <dst>/<split>.jsonl

You can also choose to flatten everything into train.jsonl via --merge-mode all-to-train
"""

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import List, Dict
from tqdm import tqdm

SPLITS = ["train", "val", "test"]

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def norm_prefix(name: str) -> str:
    # make a filesystem-friendly prefix from folder name
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in name)

def copy_or_link(src: Path, dst: Path, link: bool) -> None:
    ensure_dir(dst.parent)
    if dst.exists():
        return
    if link:
        try:
            os.symlink(src, dst)
            return
        except (OSError, NotImplementedError):
            # fallback to copy on platforms not supporting symlink
            pass
    shutil.copy2(src, dst)

def resolve_image_path(orig_path: str, src_root: Path, split: str) -> Path:
    """
    Try to resolve the actual existing image path.
    Priority:
      1) use the path as-is (absolute or relative to CWD)
      2) <src_root>/images/<split>/<basename>
      3) <src_root>/images/<basename>
    """
    p = Path(orig_path)
    if p.exists():
        return p.resolve()
    base = Path(orig_path).name
    cand1 = src_root / "images" / split / base
    if cand1.exists():
        return cand1.resolve()
    cand2 = src_root / "images" / base
    if cand2.exists():
        return cand2.resolve()
    # last resort: return as-is (will likely fail later)
    return p.resolve()

def iter_jsonl(path: Path):
    if not path.exists():
        return
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:
                continue

def prepare_dst_image_path(dst_root: Path, split: str, src_prefix: str, src_img_path: Path, dedup_cache: Dict[str, str]) -> Path:
    """
    Create a destination path like:
        <dst>/images/<split>/<prefix>_<basename>
    If file already exists with same key, reuse.
    """
    key = f"{src_prefix}|{src_img_path.name}"
    if key in dedup_cache:
        return Path(dedup_cache[key])

    base = f"{src_prefix}_{src_img_path.name}"
    dst = dst_root / "images" / split / base
    # avoid collision by adding numeric suffix
    if dst.exists():
        stem = dst.stem
        suffix = dst.suffix
        k = 1
        while True:
            cand = dst.with_name(f"{stem}_{k}{suffix}")
            if not cand.exists():
                dst = cand
                break
            k += 1

    dedup_cache[key] = dst.resolve().as_posix()
    return dst

def merge_one_split(
    sources: List[Path],
    dst_root: Path,
    split: str,
    link_images: bool,
    merge_mode: str
) -> int:
    """
    Return number of records written for this split (or into train if all-to-train).
    """
    out_split = "train" if merge_mode == "all-to-train" else split
    out_jsonl = dst_root / f"{out_split}.jsonl"
    ensure_dir(out_jsonl.parent)

    # if all-to-train and we're appending multiple splits, open append
    mode = "a" if (merge_mode == "all-to-train" and out_jsonl.exists()) else "w"
    wf = open(out_jsonl, mode, encoding="utf-8")

    total = 0
    dedup_cache: Dict[str, str] = {}  # map (prefix|basename) -> dst path

    try:
        for src in sources:
            src = src.resolve()
            prefix = norm_prefix(src.name)
            src_jsonl = src / f"{split}.jsonl"
            for obj in tqdm(list(iter_jsonl(src_jsonl)), desc=f"[{split}] {src.name}", leave=False):
                # robustly handle images field
                images = obj.get("images") or []
                new_paths: List[str] = []
                for im in images:
                    src_img = resolve_image_path(im, src, split)
                    dst_img = prepare_dst_image_path(dst_root, out_split, prefix, src_img, dedup_cache)
                    copy_or_link(src_img, dst_img, link_images)
                    new_paths.append(dst_img.resolve().as_posix())
                if new_paths:
                    obj["images"] = new_paths
                # append line
                wf.write(json.dumps(obj, ensure_ascii=False) + "\n")
                total += 1
    finally:
        wf.close()
    return total

def main():
    ap = argparse.ArgumentParser("Merge grounding datasets (JSONL + images)")
    ap.add_argument("--src", nargs="+", required=True, help="Source dataset dirs (>=2)")
    ap.add_argument("--dst", required=True, help="Destination directory")
    ap.add_argument("--link-images", action="store_true", help="Use symlink instead of copy (fall back to copy if not supported)")
    ap.add_argument("--merge-mode", choices=["by-split", "all-to-train"], default="by-split",
                    help="by-split: 合并到各自 split；all-to-train: 所有样本写入 train.jsonl")
    args = ap.parse_args()

#python /g0001sr/lzy/EasyR1/scripts/hebing.py --src /g0001sr/lzy/DIOR_RSVG_jsonl_NEW_shortcut_K/k_2p5 /g0001sr/lzy/VRSBench_json_K/k_2p5 --dst /g0001sr/lzy/datasets/all_for_train_k/k_0


    sources = [Path(p).expanduser().resolve() for p in args.src]
    dst_root = Path(args.dst).expanduser().resolve()
    ensure_dir(dst_root)

    total_all = 0
    if args.merge_mode == "by-split":
        splits = SPLITS
    else:
        # all-to-train：我们仍遍历三种 split，但全都写入 train.jsonl
        splits = SPLITS

    for split in splits:
        n = merge_one_split(sources, dst_root, split, args.link_images, args.merge_mode)
        if args.merge_mode == "by-split":
            print(f"[OK] {split}: {n} samples merged -> {dst_root / (split + '.jsonl')}")
        else:
            print(f"[OK] {split} appended to train: {n} samples")
        total_all += n

    if args.merge_mode == "all-to-train":
        print(f"\n[SUMMARY] wrote ALL samples into {dst_root / 'train.jsonl'}  total={total_all}")
    else:
        print(f"\n[SUMMARY] wrote splits under {dst_root}  total={total_all}")
        print(f"  - {dst_root/'train.jsonl'}")
        print(f"  - {dst_root/'val.jsonl'}")
        print(f"  - {dst_root/'test.jsonl'}")
    print(f"Images are under: {dst_root/'images'}")

if __name__ == "__main__":
    main()
