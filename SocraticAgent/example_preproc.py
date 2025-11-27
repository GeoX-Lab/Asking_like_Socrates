import argparse
import json
import logging
import os
import os.path as osp
import random
import uuid
from typing import Any, Dict, List, Optional

import datasets
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)


def _ensure_image_prefix(text: Optional[str]) -> str:
    """Ensures text starts with the <image> tag."""
    if not isinstance(text, str):
        return ""
    
    text = text.strip()
    if not text.lower().startswith("<image>"):
        return f"<image>\n{text}"
    return text


def build_vrsbench_vqa_records(
    data_json_path: str,
    image_root: str,
    seed: Optional[int] = None,
    skip_missing_images: bool = True,
) -> List[Dict[str, Any]]:
    """
    Parses VRSBench VQA data and formats it for training.
    """
    if seed is not None:
        random.seed(seed)

    if not osp.exists(data_json_path):
        raise FileNotFoundError(f"Data file not found: {data_json_path}")

    logger.info(f"Loading data from {data_json_path}...")
    with open(data_json_path, "r", encoding="utf-8") as f:
        all_items = json.load(f)

    # Filter for valid VQA items (2 turns, contains '[vqa]')
    subset: List[Dict[str, Any]] = []
    for item in all_items:
        conv = item.get("conversations", [])
        if len(conv) != 2:
            continue
        
        v0 = str(conv[0].get("value", "") or "")
        if "[vqa]" not in v0.lower():
            continue
        subset.append(item)

    records: List[Dict[str, Any]] = []
    
    for item in tqdm(subset, desc="Converting VRSBench-VQA", ncols=100):
        try:
            img_name = item.get("image")
            if not img_name:
                continue

            # Check image existence
            if skip_missing_images and not osp.exists(osp.join(image_root, img_name)):
                continue

            q_raw = item["conversations"][0]["value"]
            a_raw = item["conversations"][1]["value"]

            record = {
                "id": str(uuid.uuid4()),
                "query": _ensure_image_prefix(q_raw),
                "gt": (a_raw or "").strip(),
                "image": [("rgb", img_name)], # NOTE (modality, image_name)
                "image_root": {"rgb": image_root}, # NOTE (modality, image_root)
                "data_source": "VRSBench_VQA",
                "task": "vqa",
            }
            records.append(record)

        except Exception as e:
            logger.warning(f"Skipping item due to error: {e}")
            continue

    logger.info(f"Processed {len(records)} valid records.")
    return records


def to_hf_and_save(records: List[Dict[str, Any]], save_path: str) -> datasets.Dataset:
    """Converts records to a Hugging Face Dataset and saves as Parquet."""
    if not records:
        raise ValueError("No records to save.")

    data_dict = {
        "id":          [r["id"] for r in records],
        "query":       [r["query"] for r in records],
        "gt":          [r["gt"] for r in records],
        "image":       [r["image"] for r in records],
        "image_root":  [r["image_root"] for r in records],
        "data_source": [r["data_source"] for r in records],
        "task":        [r["task"] for r in records],
    }

    ds = datasets.Dataset.from_dict(data_dict)
    
    os.makedirs(osp.dirname(osp.abspath(save_path)), exist_ok=True)
    ds.to_parquet(save_path)
    logger.info(f"Saved dataset to {save_path}")
    
    return ds


def main():
    parser = argparse.ArgumentParser(description="Convert VRSBench JSON to HF Parquet format.")
    parser.add_argument("--data_path", type=str, default="SocraticAgent/demo_data/VRSBench_train.json")
    parser.add_argument("--image_root", type=str, default="SocraticAgent/demo_data/images")
    parser.add_argument("--out_path", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--keep_missing", action="store_true", help="Keep entries even if image is missing.")
    
    args = parser.parse_args()

    out_path = args.out_path
    if not out_path:
        base, _ = osp.splitext(args.data_path)
        out_path = f"{base}_convert.parquet"

    try:
        records = build_vrsbench_vqa_records(
            data_json_path=args.data_path,
            image_root=args.image_root,
            seed=args.seed,
            skip_missing_images=not args.keep_missing,
        )
        
        ds = to_hf_and_save(records, out_path)

        # Print summary
        print(f"\nDataset Info:\n{ds}")
        if len(ds) > 0:
            print("\nRandom Sample:")
            print(json.dumps(random.choice(ds), indent=2, ensure_ascii=False))

    except Exception as e:
        logger.error(f"Process failed: {e}")
        exit(1)


if __name__ == "__main__":
    main()