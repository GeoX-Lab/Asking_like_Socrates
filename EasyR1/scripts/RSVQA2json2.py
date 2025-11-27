#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
生成 train/val 两个集合（70%/30%划分）并写入 JSONL 和 CSV：
- 过滤：仅保留 yes/no 或 纯数字 的样本，且剔除答案含 'm2'
- 每图取 100 条样本，按 12 组（每组 5–12 条，总和=100）分组
- 每组生成一个多选题（A/B/C... 选项）和 problem 字段（题干+带字母选项）
- 题面正确项来自设计随机性 wants（每个选项 50% 为真）
- 计算两种奖励：
    1) realized_shortcut_reward = (exp(k * c/m) - 1) / (exp(k) - 1)  # 用当前题真实 c
    2) expected_shortcut_reward_design = sum_{c=0..m} Binom(m,c)/2^m * r(c,m)  # 只对设计随机性取期望
- 可用 --k 控制超参数，输出目录用 K 值区分：OUTDIR_BASE/RSVQA-HR-json2_K{K_TAG}
"""

import os
import re
import csv
import math
import json
import random
import argparse
from typing import Any, List, Tuple

from datasets import load_dataset
from PIL import Image
from tqdm import tqdm

# =============== 可按需修改的基础路径配置 ===============
DATA_GLOB = "/g0001sr/lzy/RSVQA-HR_qwen_finetuning/data/train-*.parquet"
OUTDIR_BASE = "/g0001sr/lzy"   # 所有输出会放在这个目录下的子目录里
PROJECT_PREFIX = "RSVQA-HR-json-K/"  # 子目录前缀

# =============== 固定配置 ===============
GROUP_SIZE_PER_IMAGE = 100
N_GROUPS_PER_IMAGE   = 12
MIN_GROUP = 5
MAX_GROUP = 12
N_IMAGES_TO_PROCESS = None   # None=全部

SEED = 42
VAL_RATIO = 0.3  # 30% 作为验证集

# =============== 规则与工具 ===============
YES_NO_SET   = {"yes", "no"}
NUMERIC_RE   = re.compile(r"^[+-]?(?:\d+(?:\.\d+)?|\.\d+)(?:[eE][+-]?\d+)?$")
IMG_NAME_RE  = re.compile(r"^img_(\d{4})\.jpg$", re.IGNORECASE)

CSV_HEADER = [
    "image_index", "group_index", "option_index",
    "question", "gold_answer", "option",
    "intended", "judged"
]

random.seed(SEED)

# ----------------- CLI -----------------
def parse_args():
    p = argparse.ArgumentParser(description="Build JSONL/CSV with A/B/C options and shortcut reward.")
    p.add_argument("--k", type=float, default=5.0, help="exponential reward parameter k (default: 5.0)")
    p.add_argument("--val_ratio", type=float, default=VAL_RATIO, help="validation split ratio (default: 0.3)")
    p.add_argument("--outdir_base", type=str, default=OUTDIR_BASE, help="base directory to write outputs")
    p.add_argument("--data_glob", type=str, default=DATA_GLOB, help="parquet glob for the input dataset")
    p.add_argument("--max_images", type=int, default=None, help="limit #images processed (None=all)")
    return p.parse_args()

def k_tag(k: float) -> str:
    # 目录中的 K 标签：整数用 K5，小数用 K5p5（把 '.' 替换为 'p'）
    if float(k).is_integer():
        return f"k{int(k)}"
    s = str(k).replace(".", "p")
    return f"k{s}"

# ----------------- CSV 辅助 -----------------
def ensure_csv_header(path: str):
    need_header = (not os.path.exists(path)) or (os.path.getsize(path) == 0)
    if need_header:
        with open(path, "w", encoding="utf-8", newline="") as f:
            csv.writer(f).writerow(CSV_HEADER)

def append_option_rows(path: str,
                       image_index: int,
                       group_index: int,
                       questions: List[str],
                       answers: List[str],
                       options: List[str],
                       wants: List[bool]):
    with open(path, "a", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        for i, (q, a, opt, w) in enumerate(zip(questions, answers, options, wants)):
            writer.writerow([
                image_index, group_index, i,
                q, a, opt,
                "correct" if w else "incorrect",
                "N/A"
            ])

# ----------------- 字段判定 -----------------
def contains_m2(ans: Any) -> bool:
    return "m2" in str(ans).lower()

def is_yes_no(ans: Any) -> bool:
    return str(ans).strip().lower() in YES_NO_SET

def is_numeric_str(ans: Any) -> bool:
    return bool(NUMERIC_RE.match(str(ans).strip()))

# ----------------- 图片保存 -----------------
def save_image_once(img: Image.Image, image_index: int, img_dir: str) -> str:
    name = f"img_{image_index:04d}.jpg"
    path = os.path.join(img_dir, name)
    img.convert("RGB").save(path, format="JPEG")
    return os.path.abspath(path)

# ----------------- 分组大小生成（总和=100） -----------------
def split_100_into_12(min_g: int, max_g: int, n_groups: int, total: int) -> List[int]:
    sizes = [random.randint(min_g, max_g) for _ in range(n_groups)]
    s = sum(sizes)
    while s != total:
        if s > total:
            idxs = [i for i, v in enumerate(sizes) if v > min_g]
            if not idxs:
                sizes = [min_g] * n_groups
                s = sum(sizes)
                continue
            i = random.choice(idxs)
            sizes[i] -= 1
            s -= 1
        else:
            idxs = [i for i, v in enumerate(sizes) if v < max_g]
            if not idxs:
                sizes = [max_g] * n_groups
                s = sum(sizes)
                continue
            i = random.choice(idxs)
            sizes[i] += 1
            s += 1
    return sizes

SIZES_PATTERN = split_100_into_12(MIN_GROUP, MAX_GROUP, N_GROUPS_PER_IMAGE, GROUP_SIZE_PER_IMAGE)

# ----------------- wants（真/假标签）生成 -----------------
_WANTS_CACHE = {}
def wants_for_len(L: int) -> List[bool]:
    if L not in _WANTS_CACHE:
        w = [bool(random.getrandbits(1)) for _ in range(L)]
        if all(w):
            w[random.randrange(L)] = False
        if not any(w):
            w[random.randrange(L)] = True
        _WANTS_CACHE[L] = w
    return _WANTS_CACHE[L][:]

def rotate(seq: List[Any], k: int) -> List[Any]:
    if not seq:
        return seq
    k %= len(seq)
    return seq[k:] + seq[:k]

def flip_yes_no(a: str) -> str:
    return "no" if a.strip().lower() == "yes" else "yes"

# ----------------- 数字扰动 -----------------
def parse_numeric(ans: str) -> Tuple[float, bool, int, bool, int]:
    s = ans.strip()
    is_sci = 'e' in s.lower()
    if is_sci:
        v = float(s)
        mantissa = s.lower().split('e')[0]
        m_dec = len(mantissa.split('.')[1]) if '.' in mantissa else 0
        return v, False, 0, True, m_dec
    v = float(s)
    if '.' in s:
        dec = len(s.split('.')[1])
        return v, False, dec, False, 0
    return v, True, 0, False, 0

def fmt_numeric(v: float, is_int: bool, dec: int, is_sci: bool, sci_dec: int) -> str:
    if is_sci:
        return f"{v:.{sci_dec}e}"
    if is_int:
        return str(int(round(v)))
    return f"{v:.{dec}f}"

def make_numeric_wrong(ans_str: str) -> str:
    v, is_int, dec, is_sci, sci_dec = parse_numeric(ans_str)
    delta = abs(v) * 0.2 if v != 0 else 1.0
    if is_int:
        delta = max(int(round(delta)), 1)
    wrong_v = v + random.choice([-1, 1]) * delta
    wrong_s = fmt_numeric(wrong_v, is_int, dec, is_sci, sci_dec)
    if wrong_s.strip() == ans_str.strip():
        wrong_v += (1 if is_int else 0.001)
        wrong_s = fmt_numeric(wrong_v, is_int, dec, is_sci, sci_dec)
    return wrong_s

# ----------------- 奖励计算 -----------------
def realized_shortcut_reward(num_correct: int, num_options: int, k: float) -> float:
    """r(c, m) = (exp(k * c/m) - 1) / (exp(k) - 1); when k=0 -> linear"""
    if num_options <= 0:
        return 0.0
    if abs(k) < 1e-8:
        # 退化为线性形式
        return num_correct / num_options
    return (math.exp(k * (num_correct / num_options)) - 1.0) / (math.exp(k) - 1.0)


def expected_shortcut_reward_design(num_options: int, k: float) -> float:
    """E[r] under design randomness; when k=0 -> linear E[c/m]=0.5"""
    m = num_options
    if m <= 0:
        return 0.0
    if abs(k) < 1e-8:
        # 期望 c/m = 0.5
        return 0.5
    denom = math.exp(k) - 1.0
    total = 0.0
    for c in range(m + 1):
        p_c = math.comb(m, c) * (0.5 ** m)
        r_c = (math.exp(k * (c / m)) - 1.0) / denom
        total += p_c * r_c
    return total
# =============== 主流程 ===============
def main():
    args = parse_args()
    K_PARAM = float(args.k)
    val_ratio = float(args.val_ratio)

    # 输出目录：按 K 值区分
    kdir = f"{PROJECT_PREFIX}/{k_tag(K_PARAM)}"
    OUTDIR = os.path.join(args.outdir_base, kdir)
    IMG_DIR = os.path.join(OUTDIR, "images")
    TRAIN_JSONL = os.path.join(OUTDIR, "train.jsonl")
    VAL_JSONL   = os.path.join(OUTDIR, "val.jsonl")
    TRAIN_CSV   = os.path.join(OUTDIR, "verify_log_train.csv")
    VAL_CSV     = os.path.join(OUTDIR, "verify_log_val.csv")

    os.makedirs(OUTDIR, exist_ok=True)
    os.makedirs(IMG_DIR, exist_ok=True)
    ensure_csv_header(TRAIN_CSV)
    ensure_csv_header(VAL_CSV)

    # 读取数据（streaming）
    ds = load_dataset("parquet", data_files={"train": args.data_glob}, split="train", streaming=True)
    it = iter(ds)

    processed_images = 0
    written_groups = 0
    total_target = args.max_images

    with open(TRAIN_JSONL, "a", encoding="utf-8") as ftrain, \
         open(VAL_JSONL,   "a", encoding="utf-8") as fval, \
         tqdm(desc="Images", unit="img", total=total_target) as pbar:

        image_index = 0
        while True:
            if total_target is not None and processed_images >= total_target:
                break

            # 抽 100 条构成一个“图块”
            block: List[dict] = []
            while len(block) < GROUP_SIZE_PER_IMAGE:
                try:
                    ex = next(it)
                except StopIteration:
                    break
                block.append(ex)
            if not block:
                break

            image_index += 1

            # 过滤样本
            fblock = []
            for ex in block:
                a_raw = str(ex.get("answer", "")).strip()
                if contains_m2(a_raw):
                    continue
                if not (is_yes_no(a_raw) or is_numeric_str(a_raw)):
                    continue
                if ex.get("image") is None:
                    continue
                fblock.append(ex)

            if len(fblock) < MIN_GROUP:
                processed_images += 1
                pbar.update(1)
                continue

            # 保存图片（取该图第一条的 image）
            img_path = save_image_once(fblock[0]["image"], image_index, IMG_DIR)

            # 分组
            sizes = rotate(SIZES_PATTERN, k=image_index % len(SIZES_PATTERN))
            groups = []
            start = 0
            for s in sizes:
                g = fblock[start:start + s]
                if not g:
                    break
                groups.append(g)
                start += s
                if start >= len(fblock):
                    break

            # train/val 归属（按图随机）
            is_val = (random.random() < val_ratio)
            fout = fval if is_val else ftrain
            csv_path = VAL_CSV if is_val else TRAIN_CSV

            # 为每组生成题目
            for gi, g in enumerate(groups):
                L = len(g)
                wants = wants_for_len(L)
                if L > 1:
                    wants = rotate(wants, k=(image_index + gi) % L)

                options, labels_idx = [], []
                q_list, a_list, opt_list, w_list = [], [], [], []

                for oi, (ex, want_true) in enumerate(zip(g, wants)):
                    q = str(ex.get("question", "")).strip()
                    a_raw = str(ex.get("answer", "")).strip()

                    if is_yes_no(a_raw):
                        ans = a_raw.strip().lower()
                        opt_ans = ans if want_true else flip_yes_no(ans)
                    else:
                        correct = a_raw.strip()
                        opt_ans = correct if want_true else make_numeric_wrong(correct)

                    opt_text = f"Q: {q} | A: {opt_ans}"
                    options.append(opt_text)
                    if want_true:
                        labels_idx.append(oi)

                    q_list.append(q)
                    a_list.append(a_raw)
                    opt_list.append(opt_text)
                    w_list.append(want_true)

                # 保证至少一个正确 & 一个错误
                if not labels_idx or len(labels_idx) == L:
                    labels_idx = [0]
                    w_list = [True] + w_list[1:]

                # 生成 A/B/C... 选项与题干
                option_labels = [chr(65 + i) for i in range(len(options))]  # 'A','B',...
                labeled_option_texts = [f"{lab}. {opt}" for lab, opt in zip(option_labels, options)]
                problem_text = "针对上面的图像，下面选项中答案能够正确回答问题的有哪些。\n" + "\n".join(labeled_option_texts)

                # 奖励计算（只考虑设计随机性）
                c = len(labels_idx)
                m = len(options)
                r_realized       = realized_shortcut_reward(c, m, K_PARAM)
                r_expect_design  = expected_shortcut_reward_design(m, K_PARAM)

                # 写 JSONL
                rec = {
                    "image": img_path,
                    "instruction": "Given the image above, which of the following options correctly answer the question?",
                    "problem": problem_text,
                    "options": dict(zip(option_labels, options)),     # {"A": "...", "B": "...", ...}
                    "labels": [option_labels[i] for i in labels_idx], # ["A","C",...]
                    "meta": {
                        "image_index": image_index,
                        "group_index": gi,
                        "group_size": m,
                        "num_correct": c,
                        "realized_shortcut_reward": r_realized,
                        "expected_shortcut_reward_design": r_expect_design,
                        "k": K_PARAM
                    }
                }
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")

                # 写 CSV（逐选项日志）
                append_option_rows(csv_path, image_index, gi, q_list, a_list, opt_list, w_list)

                written_groups += 1

            processed_images += 1
            pbar.update(1)

    print("\n=== DONE ===")
    print(f"K = {K_PARAM}")
    print(f"Images processed: {processed_images}")
    print(f"Groups written:  {written_groups}")
    print(f"Output dir:      {OUTDIR}")
    print(f"Train JSONL:     {TRAIN_JSONL}")
    print(f"Val JSONL:       {VAL_JSONL}")
    print(f"Train CSV:       {TRAIN_CSV}")
    print(f"Val CSV:         {VAL_CSV}")

if __name__ == "__main__":
    main()
