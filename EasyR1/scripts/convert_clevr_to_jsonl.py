#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
高性能 clevr_count_70k 转换器
- 本地一次性加载（streaming=False）
- 多进程导出图片
- 可选直接引用缓存路径（不重新保存图片）

依赖：pip install datasets pillow pyarrow tqdm
"""

import argparse
import json
from pathlib import Path
from io import BytesIO
from typing import List
from tqdm import tqdm
from PIL import Image as PILImage
from datasets import load_dataset, Features, Value, Sequence, Image
from multiprocessing import Pool


def build_features() -> Features:
    return Features({
        "problem": Value("string"),
        "answer": Value("string"),
        "images": Sequence(Image())
    })


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def export_one_image(img_obj, out_path: Path) -> Path:
    """导出单张图片"""
    ensure_dir(out_path.parent)

    if isinstance(img_obj, PILImage.Image):
        img_obj.save(out_path)
        return out_path.resolve()

    if isinstance(img_obj, dict):
        if img_obj.get("path", None):  # 直接拷贝
            src_path = Path(img_obj["path"])
            if src_path.exists():
                return src_path.resolve()
        if img_obj.get("bytes", None):
            bio = BytesIO(img_obj["bytes"])
            with PILImage.open(bio) as im:
                im.save(out_path)
            return out_path.resolve()

    if isinstance(img_obj, (bytes, bytearray)):
        bio = BytesIO(img_obj)
        with PILImage.open(bio) as im:
            im.save(out_path)
        return out_path.resolve()

    raise ValueError(f"Unrecognized image object type: {type(img_obj)}")


def export_images_list(images, images_out_dir: Path, base_name: str, direct_link: bool) -> List[str]:
    """导出或直接引用一条样本的所有图片"""
    out_paths: List[str] = []
    if images is None:
        return out_paths

    if not isinstance(images, (list, tuple)):
        images = [images]

    for j, img_obj in enumerate(images):
        if direct_link:
            # 直接引用 HF 缓存路径
            if isinstance(img_obj, dict) and "path" in img_obj:
                out_paths.append(Path(img_obj["path"]).resolve().as_posix())
            else:
                raise ValueError("direct_link 模式下需要 Image 字段包含 'path'")
        else:
            # 保存到输出目录
            out_path = images_out_dir / f"{base_name}_{j}.png"
            abs_path = export_one_image(img_obj, out_path)
            out_paths.append(abs_path.as_posix())
    return out_paths


def process_sample(args):
    idx, ex, images_out_dir, direct_link = args
    problem = ex.get("problem", "")
    answer = str(ex.get("answer", ""))
    imgs = ex.get("images", None)
    img_paths = export_images_list(imgs, images_out_dir, base_name=str(idx), direct_link=direct_link)
    return {
        "problem": problem,
        "answer": answer,
        "images": img_paths,
    }


from tqdm import tqdm

def convert_split(src_dir: Path, out_dir: Path, split: str, out_jsonl_name: str, direct_link: bool, num_proc: int):
    features = build_features()
    data_files = str(src_dir / "data" / f"{split}-*.parquet")

    print(f"[INFO] Loading {split} split from {data_files} ...")
    ds = load_dataset(
        "parquet",
        data_files={split: data_files},
        split=split,
        features=features,
        streaming=False
    )

    jsonl_path = out_dir / out_jsonl_name
    ensure_dir(jsonl_path.parent)
    images_out_dir = out_dir / "images" / split
    ensure_dir(images_out_dir)

    print(f"[INFO] Exporting {split} split with {num_proc} processes ...")
    tasks = ((idx, ex, images_out_dir, direct_link) for idx, ex in enumerate(ds))

    results = []
    with Pool(processes=num_proc) as pool:
        for rec in tqdm(pool.imap(process_sample, tasks), total=len(ds), desc=f"Processing {split}"):
            results.append(rec)

    with open(jsonl_path, "w", encoding="utf-8") as f:
        for rec in results:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"[OK] {split}: wrote {len(results)} samples to {jsonl_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True, help="本地 clevr_count_70k 根目录（包含 data/ 与 dataset_info.json）")
    parser.add_argument("--out", required=True, help="输出目录")
    parser.add_argument("--direct-link", action="store_true", help="直接引用 HF 缓存路径，不重新保存图片")
    parser.add_argument("--num-proc", type=int, default=8, help="并行进程数")
    args = parser.parse_args()

    src_dir = Path(args.src).expanduser().resolve()
    out_dir = Path(args.out).expanduser().resolve()

    convert_split(src_dir, out_dir, split="train", out_jsonl_name="train.jsonl",
                  direct_link=args.direct_link, num_proc=args.num_proc)
    convert_split(src_dir, out_dir, split="test", out_jsonl_name="val.jsonl",
                  direct_link=args.direct_link, num_proc=args.num_proc)

    print("\nDone. 使用方法示例：")
    print(f"  data.train_files={out_dir.as_posix()}/train.jsonl \\")
    print(f"  data.val_files={out_dir.as_posix()}/val.jsonl \\")
    print("  data.image_dir=null")


if __name__ == "__main__":
    main()

