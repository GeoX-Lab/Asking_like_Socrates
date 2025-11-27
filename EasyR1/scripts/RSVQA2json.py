#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
VQA multi-select option builder via vLLM (DeepSeek-R1-Distill-Qwen-7B)
+ Per-option CSV logging (no external verifier)
+ Filter: keep ONLY questions whose gold answers are strictly Yes/No OR pure numeric

Changes vs previous:
- Keep QA pairs whose gold answer is exactly 'yes' or 'no' (case-insensitive), OR a pure number
  (int/float/scientific notation; no units/extra chars).
- Numeric answers get their own strict prompt template:
    want_true=True  -> <answer>The answer is {NUM}.</answer>
    want_true=False -> <answer>The answer is not {NUM}.</answer>
  using the exact numeric string from Gold Answer, with no other numbers/units.
- Single-process vLLM batch generation; CSV judged='N/A'.
"""

import os, io, json, random, re, hashlib, csv
from typing import List, Dict, Any, Optional, Tuple
from statistics import mean
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm

# vLLM
from vllm import LLM, SamplingParams

# ================= Fixed config =================
DATA_GLOB = "/g0001sr/lzy/RSVQA-HR_qwen_finetuning/data/train-*.parquet"
GROUP_SIZE_PER_IMAGE = 100
N_GROUPS_PER_IMAGE = 12
MIN_GROUP = 5
MAX_GROUP = 12
N_IMAGES_TO_PROCESS = None
SEED = 42

OUTDIR = "/g0001sr/lzy/RSVQA-HR-json"
IMG_DIR = os.path.join(OUTDIR, "images")
JSONL_PATH = os.path.join(OUTDIR, "train.jsonl")
CSV_PATH = os.path.join(OUTDIR, "verify_log.csv")  # per-option log (judged fixed to N/A)

# ===== Generator model (via vLLM) =====
DEEPSEEK_ID = "/g0001sr/model_weights/Qwen2.5-VL-7B-Instruct"
GEN_MAX_NEW_TOKENS = 256
GEN_TEMPERATURE = 0.9
GEN_TOP_P = 0.95

# vLLM params
VLLM_TP = 4             # None -> auto; or int (2/4/...) for tensor parallel
VLLM_PIPELINE_PARALLEL = 1
VLLM_MAX_MODEL_LEN = 4096
VLLM_GPU_MEMORY_UTILIZATION = 0.90
VLLM_MAX_NUM_SEQS = None

random.seed(SEED)
os.makedirs(OUTDIR, exist_ok=True)
os.makedirs(IMG_DIR, exist_ok=True)

# ================= Helpers (shared) =================
answer_tag_pat = re.compile(r"<\s*answer\s*>([\s\S]*?)<\s*\/\s*answer\s*>", re.IGNORECASE)
IMG_NAME_RE = re.compile(r"^img_(\d{4})\.jpg$")

YES_NO_SET = {"yes", "no"}
# pure number: int/float/.5/5./scientific notation; optional sign
_NUMERIC_RE = re.compile(r"^[+-]?(?:\d+(?:\.\d+)?|\.\d+)(?:[eE][+-]?\d+)?$")

def is_yes_no_answer(ans: Any) -> bool:
    """Keep only 'yes' or 'no' (case-insensitive), exact tokens."""
    s = str(ans).strip().lower()
    return s in YES_NO_SET

def is_numeric_answer(ans: Any) -> bool:
    """Keep only pure numeric strings: int/float/scientific; no units or extra chars."""
    s = str(ans).strip()
    return bool(_NUMERIC_RE.match(s))

def uid_from_img(img: Image.Image) -> str:
    path = getattr(img, "filename", None)
    if path: return os.path.basename(path)
    buf = io.BytesIO(); img.convert("RGB").save(buf, format="PNG")
    return "sha1:" + hashlib.sha1(buf.getvalue()).hexdigest()

def save_image_once(img: Image.Image, image_index: int) -> str:
    name = f"img_{image_index:04d}.jpg"
    path = os.path.join(IMG_DIR, name)
    img.convert("RGB").save(path, format="JPEG")
    return os.path.abspath(path)

def build_prompt(question: str, answer: str, want_true: bool) -> str:
    base_rule = (
        "You are a generator of multiple-choice option statements for visual question answering (VQA).\n"
        "I will give you the Question of VQA and the original Gold Answer of this question.Please produce ONE concise English statement (<= 25 words). "
        "Don't output any additional words."
    )
    if want_true:
        rule = (
            #f'For this option, you need to use the information provided by the Question and the Gold Answer to create a correct one. Your response should be included in <answer></answer>.\n'
            f'For this option, you need to use the information provided by Question and Gold Answer to create an option that is consistent with the original information. Your reply should be included in <answer></answer>.'
                "Examples:\n"
                # 你提供的例子（no + 正确）
                "Question1: Is there a commercial building?\n"
                "Gold Answer1: no\n"
                "Your Answer1:<answer>There isn't any commercial building.</answer>\n\n"
                "Question2: Is a nature reserve present?\n"
                "Gold Answer2: yes\n"
                "Your Answer2<answer>A nature reserve is present in the image.</answer>\n\n"
        )
    else:
        #rule = 'For this option, you need to use the information provided by the Question and the Gold Answer to create an incorrect option. Your response should be included in <answer></answer>.\n'
        rule = (
            'For this option, you need to use the information provided by Question and Gold Answer to create an option that contradicts the original information. Your reply should be included in <answer></answer>'
            "Examples:\n"
            "Question1: Is a lighthouse present?\n"
            "Gold Answer1: no\n"
            "Your Answer1:<answer>A lighthouse is present in the image.</answer>\n\n"
            "Question2: Is a bridge present?\n"
            "Gold Answer2: yes\n"
            "Your Answer2:<answer>No bridge is visible in the scene.</answer>\n\n"
            )

    q = question.strip()
    a = str(answer).strip()
    prompt = (
        f"{base_rule}\n"
        f"Question: {q}\n"
        f"Gold Answer: {a}\n"
        f"{rule}\n"
    )
    return prompt

def extract_answer(text: str) -> str:
    matches = list(answer_tag_pat.finditer(text))
    if matches:
        return matches[-1].group(1).strip()
    for line in reversed(text.strip().splitlines()):
        if line.strip():
            return re.sub(r"</?answer>", "", line.strip(), flags=re.I)
    return "Unparseable output"

def cleanup_answer(s: str) -> str:
    s = re.sub(r"</?\s*answer\s*>", "", s, flags=re.IGNORECASE).strip()
    s = s.strip().strip('"').strip("'").strip('`')
    s = re.sub(r"\s+", " ", s)
    return s

def split_100_into_20_once(min_g: int, max_g: int, n_groups: int, total: int) -> List[int]:
    n, target = n_groups, total
    sizes = [random.randint(min_g, max_g) for _ in range(n)]
    s = sum(sizes)
    while s != target:
        if s > target:
            idxs = [i for i,v in enumerate(sizes) if v > min_g]
            if not idxs: sizes = [min_g]*n; s = sum(sizes); continue
            i = random.choice(idxs); sizes[i]-=1; s-=1
        else:
            idxs = [i for i,v in enumerate(sizes) if v < max_g]
            if not idxs: sizes = [max_g]*n; s = sum(sizes); continue
            i = random.choice(idxs); sizes[i]+=1; s+=1
    return sizes

SIZES_PATTERN = split_100_into_20_once(MIN_GROUP, MAX_GROUP, N_GROUPS_PER_IMAGE, GROUP_SIZE_PER_IMAGE)

WANTS_TEMPLATES: Dict[int, List[bool]] = {}
def wants_for_len(L: int) -> List[bool]:
    if L not in WANTS_TEMPLATES:
        w = [bool(random.getrandbits(1)) for _ in range(L)]
        if all(w):        w[random.randrange(L)] = False
        elif not any(w):  w[random.randrange(L)] = True
        WANTS_TEMPLATES[L] = w
    return WANTS_TEMPLATES[L]

def rotate_pattern(seq: List[Any], k: int) -> List[Any]:
    if not seq: return seq
    k %= len(seq);  return seq[k:] + seq[:k]

def load_done_image_indices(img_dir: str) -> set:
    done = set()
    try:
        for fn in os.listdir(img_dir):
            m = IMG_NAME_RE.match(fn)
            if m:
                done.add(int(m.group(1)))
    except FileNotFoundError:
        pass
    return done

# ================= CSV helper =================
CSV_HEADER = [
    "image_index", "group_index", "option_index",
    "question", "answer", "option",
    "intended",       # "correct" or "incorrect" (from wants)
    "judged"          # fixed "N/A"
]

def ensure_csv_header(path: str):
    need_header = (not os.path.exists(path)) or (os.path.getsize(path) == 0)
    if need_header:
        with open(path, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(CSV_HEADER)

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
                image_index,
                group_index,
                i,
                q,
                a,
                opt,
                "correct" if w else "incorrect",
                "N/A"
            ])

# ================= vLLM helpers =================
def build_vllm(engine_model: str) -> LLM:
    llm = LLM(
        model=engine_model,
        tensor_parallel_size=(VLLM_TP or 1),
        pipeline_parallel_size=VLLM_PIPELINE_PARALLEL,
        max_model_len=VLLM_MAX_MODEL_LEN,
        gpu_memory_utilization=VLLM_GPU_MEMORY_UTILIZATION,
        max_num_seqs=VLLM_MAX_NUM_SEQS
    )
    return llm

def gen_with_vllm(llm: LLM, prompts: List[str]) -> List[str]:
    sampling = SamplingParams(
        temperature=GEN_TEMPERATURE,
        top_p=GEN_TOP_P,
        max_tokens=GEN_MAX_NEW_TOKENS,
        n=1,
        stop=None,
    )
    outputs = llm.generate(prompts, sampling)
    texts = []
    for out in outputs:
        if out.outputs:
            texts.append(out.outputs[0].text)
        else:
            texts.append("")
    return texts

# ================= Main =================
def main():
    # resume: read finished image indices
    done_indices = load_done_image_indices(IMG_DIR)
    done_count = len(done_indices)
    if done_count:
        print(f"[Resume] Found {done_count} finished images in {IMG_DIR}. Will skip those indices.")

    # vLLM engine
    print(f"[vLLM] Loading model: {DEEPSEEK_ID}")
    llm = build_vllm(DEEPSEEK_ID)
    print("[vLLM] Ready.")

    # CSV init
    ensure_csv_header(CSV_PATH)

    # dataset stream
    ds = load_dataset("parquet", data_files={"train": DATA_GLOB}, split="train", streaming=True)
    it = iter(ds)

    image_index = 0
    written_groups_total = 0
    processed_images = 0
    skipped_not_yesno_numeric_total = 0

    total_target = N_IMAGES_TO_PROCESS if N_IMAGES_TO_PROCESS is not None else None
    initial_done = min(done_count, total_target) if total_target is not None else done_count

    with open(JSONL_PATH, "a", encoding="utf-8") as fout, \
         tqdm(total=total_target, initial=initial_done, desc="Images", unit="img") as pbar:

        processed_images = done_count

        while True:
            if total_target is not None and processed_images >= total_target:
                break

            block = _read_one_image_block(it)
            if not block:
                break
            image_index += 1

            # already done? skip entire image
            if image_index in done_indices:
                pbar.update(1)
                processed_images += 1
                continue

            # --- Filter: keep only yes/no OR numeric answers ---
            before = len(block)
            block = [ex for ex in block if (is_yes_no_answer(ex.get("answer", "")) or
                                            is_numeric_answer(ex.get("answer", "")))]
            after = len(block)
            skipped_not_yesno_numeric_total += (before - after)

            # if nothing left after filtering, skip this image
            if not block:
                pbar.update(1)
                processed_images += 1
                continue

            img_path = save_image_once(block[0]["image"], image_index)

            # group with precomputed size pattern; partial groups are fine
            sizes = rotate_pattern(SIZES_PATTERN, k=image_index % len(SIZES_PATTERN))
            groups = []
            start = 0
            for s in sizes:
                g = block[start:start+s]
                if g:
                    groups.append(g)
                start += s
                if start >= len(block):
                    break  # no more samples in this image after filtering

            # build prompts across all groups in this image
            all_prompts: List[str] = []
            index_spans: List[Tuple[int, int]] = []  # (start,end) indices in all_prompts
            wants_per_group: List[List[bool]] = []
            qa_per_group: List[Tuple[List[str], List[str]]] = []

            for gi, g in enumerate(groups):
                L = len(g)
                wants = rotate_pattern(wants_for_len(L), k=image_index % max(L, 1)) if L > 1 else wants_for_len(L)
                q_list = [str(qa["question"]).strip() for qa in g]
                a_list = [str(qa["answer"]).strip()   for qa in g]

                start_idx = len(all_prompts)
                for q, a, w in zip(q_list, a_list, wants):
                    all_prompts.append(build_prompt(q, a, w))
                end_idx = len(all_prompts)

                index_spans.append((start_idx, end_idx))
                wants_per_group.append(wants)
                qa_per_group.append((q_list, a_list))

            with tqdm(total=len(groups), desc=f"Image {image_index:04d} groups", leave=False, unit="group") as gpbar:
                # generate once for all prompts in this image
                raw_texts = gen_with_vllm(llm, all_prompts)

                # unpack and write
                for gi, (span, wants) in enumerate(zip(index_spans, wants_per_group)):
                    s, e = span
                    texts = raw_texts[s:e]
                    options = [cleanup_answer(extract_answer(t)) if t else "Unparseable output" for t in texts]

                    q_list, a_list = qa_per_group[gi]

                    # CSV: judged=N/A
                    append_option_rows(
                        CSV_PATH,
                        image_index=image_index,
                        group_index=gi,
                        questions=q_list,
                        answers=a_list,
                        options=options,
                        wants=wants
                    )

                    # basic group constraints
                    if len(options) < MIN_GROUP:
                        gpbar.update(1)
                        continue
                    labels = [i for i, w in enumerate(wants) if w]
                    has_true = len(labels) > 0
                    has_false = len(labels) < len(options)
                    if not (has_true and has_false):
                        gpbar.update(1)
                        continue

                    rec = {
                        "image": img_path,
                        "options": options,
                        "labels": labels,
                        "meta": {
                            "image_index": image_index,
                            "group_index": gi,
                            "group_size": len(options),
                            "filtered": {
                                "original": len(options),
                                "kept": len(options)
                            }
                        }
                    }
                    fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    written_groups_total += 1
                    gpbar.update(1)

            kept_count = sum(len(g) for g in groups)
            if groups:
                g_lens = [len(g) for g in groups]
                print(f"[Image {image_index:04d}] kept {kept_count} QA after yes/no+numeric filter; "
                      f"groups mean={mean(g_lens):.2f} min={min(g_lens)} max={max(g_lens)} -> groups {len(groups)}")
            else:
                print(f"[Image {image_index:04d}] no groups after filtering.")

            pbar.update(1)
            processed_images += 1

    print("\n=== DONE ===")
    print(f"images processed (this run + previous): {processed_images}")
    print(f"groups written (this run): {written_groups_total}")
    print(f"skipped QA due to not yes/no & not numeric: {skipped_not_yesno_numeric_total}")
    print(f"images dir: {os.path.abspath(IMG_DIR)}")
    print(f"train.jsonl (appended): {os.path.abspath(JSONL_PATH)}")
    print(f"per-option csv: {os.path.abspath(CSV_PATH)}")

def _read_one_image_block(it):
    """Read exactly GROUP_SIZE_PER_IMAGE samples as one image block."""
    block = []
    for _ in range(GROUP_SIZE_PER_IMAGE):
        try:
            ex = next(it)
        except StopIteration:
            break
        block.append(ex)
    if not block:
        return None
    return block

if __name__ == "__main__":
    main()
