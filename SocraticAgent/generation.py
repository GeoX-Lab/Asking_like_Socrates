import re
import os
from PIL import Image
from tqdm import tqdm
import json
import random
import datasets
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import uuid
import argparse

import sys
sys.path.append('.')
from utils import APIModel


# =========================
# Regex & Constants
# =========================

FINAL_PREFIX_EN = "[Final Answer]:"
FINAL_PREFIX = FINAL_PREFIX_EN

FINALLINE_RE_EN = re.compile(
    r"^\s*\[Final\s+Answer\]\s*:\s*(.*)\Z",
    re.IGNORECASE | re.MULTILINE | re.DOTALL
)
FINALLINE_RE_LIST = [FINALLINE_RE_EN]

THINK_RE = re.compile(r"<thinking>\s*([\s\S]*?)\s*</thinking>", re.IGNORECASE)
QUESTION_RE = re.compile(r"<question>\s*([\s\S]*?)\s*</question>", re.IGNORECASE)


# =========================
# Small helpers
# =========================

def extract_final_from_text(text: str) -> Optional[str]:
    if not isinstance(text, str):
        return None
    for pat in FINALLINE_RE_LIST:
        m = pat.search(text)
        if m:
            return m.group(1).strip()
    return None


def _parse_verdict(text: str) -> Tuple[Optional[bool], Optional[str]]:
    if not text:
        return (None, None)
    t = text.strip()
    up = t.upper()
    if up == "ACCEPT":
        return (True, None)
    if up.startswith("REJECT"):
        reason = None
        if ":" in t:
            reason = t.split(":", 1)[1].strip()
        return (False, reason)
    return (None, None)


def dbg(msg: str, debug: bool):
    if debug:
        print(msg)


def dbg_header(title: str, debug: bool, sep: str = "—"):
    if debug:
        line = sep * 12
        print(f"\n{line} {title} {line}")


# =========================
# Prompts (reasoner / perception / verifier)
# =========================

def build_reasoner_system_prompt(max_loop: int) -> str:
    return f"""
You are a reasoning model that follows a **Plan–Integrate–Decide** paradigm, collaborating with a **weak-perception visual model** to complete **general remote sensing tasks** (such as classification/attribute recognition, localization/counting, relation/change detection, and VQA).  
The perception model can only answer **very simple, atomic visual facts** and cannot perform reasoning.  
Therefore, you must decompose the perception process into a **coarse-to-fine sequence of steps**, simulating how humans visually interpret remote sensing imagery.

[Coarse-to-Fine Perception Chain]
1) **Global Observation Stage (Overall Understanding):**
   - Begin with a **broad, holistic examination** of the entire image, forming an initial impression of its overall layout — main land-cover types, spatial organization, scene functionality, distribution of major objects, and possible visual interferences (e.g., shadows, fog, noise, or occlusion).
   - While questions at this stage should remain **broad, general, and high-level**, they must be **context-aware** — i.e., lightly tailored to the task/query so they inform later reasoning for this specific problem.

2) **Focused and Detailed Observation Stage (Targeted Analysis):**  
   - After forming a general understanding of the scene, use the task objective (query) and global observations to **focus attention** on potentially relevant local regions or objects.  
   - Naturally shift attention from overall impressions to specific, task-relevant areas, similar to how humans visually focus.  
   - Ask more detailed and targeted questions, typically focusing on:  
     • Local details (shape, texture, boundaries, orientation, color features, etc.);  
     • Relationships and differences (changes, similarities, transitions between regions, etc.);  
     • Task-critical elements (e.g., presence, quantity, or arrangement of specific targets).

3) **Integration and Verification Stage:**  
   - Integrate the facts collected from the global and detailed observation stages into a consistent intermediate conclusion.  
   - If contradictions or uncertainties remain, ask verification questions.  
   - Ensure that the reasoning covers all key regions and that the logic is consistent.

4) **Final Review and Confirmation Stage:**  
   - Before giving the final answer, perform a quick overall review of the image to confirm that no small anomalies, marginal areas, or potential clues have been overlooked.  
   - Check whether the final answer meets the query’s requirements regarding format, length, and structure.  
   - The final answer must only output the direct answer to the query itself, such as “Yes/No”, a specific number, or a concise conclusion. Do not include any explanations, reasoning, or additional commentary.
   - If necessary, ask one final targeted question for confirmation.

[Questioning and Iteration Constraints]
- **Never** forward the user’s original query directly to the perception model; each question must concern **only one atomic visual fact**.  
- Each new question should provide **maximum information gain** and **must not repeat** previous questions (avoid paraphrasing).
- You have {max_loop - 1} questioning rounds available: the early rounds focus on global perception, the middle rounds gather key evidence, and the final rounds perform verification questioning.

[Output Format (Strict Requirements)]
- **If further questioning is needed:**  
  Start with `<thinking>...</thinking>` (briefly explain the reasoning and purpose of the next question),  
  then output **only one** `<question>...</question>`.  
- **If ready to give the final answer:**  
  Start with `<thinking>...</thinking>` (summarize key evidence and note that final checks have been completed),  
  then output `{FINAL_PREFIX_EN} ...`.  
- Each round must **begin** with `<thinking>...</thinking>` and be followed by **exactly one** of the two options:  
  `<question>...</question>` or `{FINAL_PREFIX_EN} ...`.  
  No other content is allowed.  
- Inside `<thinking>`, do **not** mention external entities such as “the perception model,” “user,” or “conversation.”  
- Use **English** for internal reasoning and questioning, but ensure that the **final answer matches the input query’s language**.
"""


P_SYSTEM_PROMPT = """
You are an image interpretation expert collaborating with a **reasoning model that has very weak logical ability**. 
Together, through multi-turn dialogue, you will complete **general remote sensing tasks** (classification/attribute, localization/counting, relation/change analysis, VQA, etc.).

The reasoning model can **only understand the textual descriptions** of your perception results — it **cannot see the image directly**. 
Therefore, you must respond to each of its questions about the image **accurately and completely**, without adding any information that is irrelevant to the question.

Your tone should resemble a **natural inner monologue** of a person carefully observing an image.  
Always begin your response with: “Let’s look at the image,” and then continue with your detailed observation.
"""

VERIFY_SYSTEM_PROMPT = """
You are a strict answer evaluator. Given a Query, Answer, and GT, output only:
1) "ACCEPT"
2) "REJECT: <brief reason>"

The reason must be actionable and must not leak or restate GT/Answer details.
"""


# =========================
# History builders
# =========================

def build_reasoner_history(chat_history: List[Dict[str, Any]]) -> str:
    lines: List[str] = []
    for turn in chat_history:
        if turn.get("R_response"):
            lines.append(turn["R_response"])
        if turn.get("P_response"):
            lines.append(str(turn["P_response"]))
    return "\n".join(lines)


def build_self_talk_merged(segments: List[str], final_answer_text: Optional[str]) -> str:
    header_en = "Alright, I will reason in a self Q&A style and give the final reply."
    think_block = "".join(segments) if segments else ""
    merged = f"<think>{header_en}{think_block}</think>"
    if final_answer_text:
        merged += f"{final_answer_text}"
    return merged


def ensure_image_marker(q: str) -> str:
    if "<image>" not in q:
        return ("<image>" + q.strip()).strip()
    return q.strip()


def rewrite_query_with_reasoner(raw_query: str) -> str:
    sys_prompt = (
        "You are an instruction rewriter. The input is a user query, which may be a question or a command. "
        "Rewrite the query to be clearer, more fluent, and easier to understand **without changing its meaning**. "
        "If the original query contains requirements about the answer format (e.g., JSON, table, list), keep them exactly. "
        "Output only the rewritten query, with no explanations or extra text."
    )

    user_query = "Original query: " + raw_query
    try:
        rewritten = APIModel("gpt-5-mini", sys_prompt).get_response(user_query)
        if isinstance(rewritten, tuple):
            # compatible with (thinking, response)
            rewritten = rewritten[1]
    except Exception:
        rewritten = raw_query
    rewritten = ensure_image_marker(str(rewritten))
    return rewritten


# =========================
# Core loop
# =========================

def verify_with_model(
    verify_model: APIModel,
    query: str,
    answer_text: str,
    gt_text: str,
    verify_inst: Optional[str],
) -> Tuple[Optional[bool], Optional[str], Optional[str], Optional[str]]:
    if gt_text is None:
        return (None, None, None, None)

    ds_inst = (verify_inst or "").strip()
    header = "Decide whether the Answer is acceptable."
    v_query = ((ds_inst + "\n\n") if ds_inst else "") + f"{header}\nQuery: {query}\nAnswer: {answer_text}\nGT: {gt_text}\n"

    try:
        v_raw = verify_model.get_response(v_query)
        v_raw_str = "" if v_raw is None else str(v_raw).strip()
        accepted, reason = _parse_verdict(v_raw_str)
        return (accepted, reason, v_query, v_raw_str)
    except Exception as e:
        return (None, None, v_query, f"ERROR: {e}")


def socraticLoop(
    r_model: APIModel,
    p_model: APIModel,
    data_sample: Dict[str, Any],
    max_loop: int,
    verify_model: Optional[APIModel] = None,
    verify_inst: Optional[str] = None,
    debug: bool = False
) -> Dict[str, Any]:
    result = {"success": False, "final_answer": None, "chat_history": [], "self_talk_merged": "", "error": None}
    chat_history: List[Dict[str, Any]] = result["chat_history"]
    gt_text = data_sample.get("gt")

    base_prompt_for_R = (
        "# User Query:\n" + f"{data_sample['query']}\n\n" +
        "# Image Metadata:\n" + f"{data_sample['meta_information']}\n"
    )

    story_segments: List[str] = []

    dbg_header("socraticLoop START", debug)
    dbg(f"Query: {data_sample['query']}", debug)
    dbg(f"Metadata: {data_sample['meta_information']}", debug)

    for _round in range(1, max_loop + 1):
        dbg_header(f"Round #{_round}", debug)

        r_query = base_prompt_for_R if not chat_history else base_prompt_for_R + "\n\n" + build_reasoner_history(chat_history)
        dbg("Reasoner Input (truncated 600):", debug)
        dbg(r_query[:600] + ("..." if len(r_query) > 600 else ""), debug)

        try:
            r_response = r_model.get_response(r_query)
        except Exception as e:
            dbg(f"[R ERROR] {e}", debug)
            result["error"] = "reasoner_api_error"
            result["self_talk_merged"] = build_self_talk_merged(story_segments, None)
            return result

        dbg("Reasoner Output (truncated 600):", debug)
        dbg((r_response or "")[:600] + ("..." if r_response and len(r_response) > 600 else ""), debug)

        thinking_match = THINK_RE.search(r_response or "")
        if not thinking_match:
            chat_history.append({"round": _round, "R_response": r_response})
            result["error"] = "reasoner_missing_thinking"
            dbg("[FORMAT] Missing <thinking>", debug)
            result["self_talk_merged"] = build_self_talk_merged(story_segments, None)
            return result

        thinking_text = thinking_match.group(1).strip()
        if _round > 1:
            thinking_text = "Based on the observed details, " + thinking_text
        story_segments.append(thinking_text)

        cur_questions = [m.strip() for m in QUESTION_RE.findall(r_response or "")]
        final_answer_text = extract_final_from_text(r_response or "")
        has_q = len(cur_questions) == 1
        has_final_line = final_answer_text is not None

        if (has_q and has_final_line) or (not has_q and not has_final_line):
            chat_history.append({"round": _round, "R_response": r_response})
            result["error"] = "output_format_error"
            dbg("[FORMAT] Must contain exactly one of <question> or final-line", debug)
            result["self_talk_merged"] = build_self_talk_merged(story_segments, None)
            return result

        if has_q:
            question_text = re.sub(r"\s+", " ", cur_questions[0].strip())
            story_segments.append(question_text)

            try:
                vlm_images = [(pil, "jpeg") for _, pil in data_sample["image"]]
                p_response = p_model.get_response(question_text, images=vlm_images)
                if not isinstance(p_response, str):
                    p_response = p_response[-1]
            except Exception as e:
                chat_history.append({"round": _round, "R_response": r_response})
                result["error"] = "perception_api_error"
                dbg(f"[P ERROR] {e}", debug)
                result["self_talk_merged"] = build_self_talk_merged(story_segments, None)
                return result

            dbg("Perception Output (truncated 400):", debug)
            dbg((str(p_response) or "")[:400] + ("..." if p_response and len(str(p_response)) > 400 else ""), debug)

            story_segments.append(str(p_response))
            chat_history.append({"round": _round, "R_response": r_response, "P_response": p_response})
            continue

        # final
        if has_final_line:
            dbg(f"Detected Final Line: {final_answer_text}", debug)

            v_query = v_raw = v_reason = None
            is_accept: Optional[bool] = None

            if verify_model is not None and gt_text is not None:
                is_accept, v_reason, v_query, v_raw = verify_with_model(
                    verify_model, data_sample["query"], final_answer_text, gt_text, verify_inst
                )
                dbg(f"Verifier -> accepted={is_accept}, reason={v_reason}", debug)
            else:
                dbg("Verifier disabled or GT missing", debug)

            chat_history.append({
                "round": _round,
                "R_response": r_response,
                "V_decision": "ACCEPT" if is_accept else ("SKIPPED" if (verify_model is None or gt_text is None) else "REJECT"),
                "V_reason": v_reason
            })
            result["final_answer"] = final_answer_text
            result["self_talk_merged"] = build_self_talk_merged(story_segments, final_answer_text)
            result["success"] = (is_accept is True) or (verify_model is None) or (gt_text is None)
            dbg_header("ACCEPTED or SKIPPED VERIFY" if result["success"] else "REJECTED BY VERIFY", debug, "=")
            return result

    result["error"] = "max_rounds_exceeded"
    result["self_talk_merged"] = build_self_talk_merged(story_segments, None)
    dbg_header("MAX ROUNDS EXCEEDED", debug, "=")
    return result


# =========================
# Image/io helpers
# =========================

def build_input_images(item: Dict[str, Any]) -> List[Tuple[str, Image.Image]]:
    """
    Build input images for the VLM.

    Required fields in each row of the parquet (per item):
    - item["image"]: List[List[str, str]] or List[Tuple[str, str]]
        Example: [["rgb", "xxx.jpg"], ["sar", "yyy.png"]]
        Each inner pair is [modality, filename].
    - item["image_root"]: Dict[str, str]
        Example: {"rgb": "/path/to/rgb_root", "sar": "/path/to/sar_root"}

    If these keys are missing, a KeyError will be raised.
    """
    imgs: List[Tuple[str, Image.Image]] = []

    image_list = item["image"]          # required
    image_root = item["image_root"]    # required

    if isinstance(image_list, list):
        for pair in image_list:
            if isinstance(pair, (list, tuple)) and len(pair) == 2:
                modality, filename = pair
                root = image_root[modality] if modality in image_root else ""
                img_path = os.path.join(root, filename)
                if os.path.exists(img_path):
                    pil = Image.open(img_path).convert("RGB")
                    imgs.append((modality, pil))
    return imgs


def map_modality_name(modality_raw: str) -> str:
    key = (modality_raw or "").strip().lower()
    if key in {"rgb", "vis", "optical"}:
        return "RGB / Visible"
    if key in {"inf", "ir", "infrared"}:
        return "Infrared"
    if key in {"sar"}:
        return "Synthetic Aperture Radar (SAR)"
    return modality_raw


def get_image_meta(img_meta_pre: str, input_images: List[Tuple[str, Image.Image]]) -> str:
    """
    Always append basic image metadata (modality + size) for all images.

    - img_meta_pre: user-provided prefix text for image metadata.
    - input_images: list of (modality, PIL.Image).
    """
    meta = img_meta_pre
    if not input_images:
        return meta
    if len(input_images) == 1:
        modality, pil = input_images[0]
        modality_disp = map_modality_name(modality)
        w, h = pil.size
        meta += f"- Modality: {modality_disp} (size: {w}x{h})"
        return meta
    for idx, (modality, pil) in enumerate(input_images, start=1):
        modality_disp = map_modality_name(modality)
        w, h = pil.size
        meta += f"- Image {idx}: Modality: {modality_disp}, size: {w}x{h}\n"
    return meta


# =========================
# Checkpointing helpers
# =========================

def get_item_id(item: Dict[str, Any]) -> str:
    """
    Try to extract a stable id from the item:
    - item["id"] (if present)
    - item["uid"] (if present)
    - item["image_id"] (if present)
    Otherwise, generate a new UUID4 string.
    """
    rid = item.get("id") or item.get("uid") or item.get("image_id")
    if rid is None:
        rid = str(uuid.uuid4())
    return str(rid)


def load_processed_ids_from_jsonl(jsonl_path: str) -> set:
    ids = set()
    if not os.path.exists(jsonl_path):
        return ids
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                rid = rec.get("id")
                if rid is not None:
                    ids.add(str(rid))
            except Exception:
                continue
    return ids


def append_jsonl(jsonl_path: str, record: Dict[str, Any]):
    os.makedirs(os.path.dirname(jsonl_path), exist_ok=True)
    with open(jsonl_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


# =========================
# Per-item worker
# =========================

def process_item(
    item: Dict[str, Any],
    rid: str,
    r_model: APIModel,
    p_model: APIModel,
    verify_model: Optional[APIModel],
    verify_inst: Optional[str],
    img_meta_pre: str,
    max_loop: int,
    out_jsonl: str,
    debug: bool
) -> Dict[str, Any]:
    """
    Expected parquet row format (per item):

    Required fields:
    - item["query"]: str
        The original user query / instruction for the remote sensing task.
    - item["image"]: List[List[str, str]] or List[Tuple[str, str]]
        Example: [["rgb", "img_0001.jpg"], ["sar", "img_0001_sar.png"]]
        Each pair is [modality, filename].
    - item["image_root"]: Dict[str, str]
        Example: {"rgb": "/data/rgb", "sar": "/data/sar"}

    Optional fields:
    - item["gt"]: Any
        Ground-truth answer, used for automatic verification (if provided and verify_inst is set).
    - item["task"]: str
        Task name or type.
    - item["data_source"]: str
        Dataset or source identifier.
    - item["id"] / item["uid"] / item["image_id"]: Any
        Used as unique identifier (if none exists, a UUID4 will be generated).
    """
    try:
        # Required: these will raise KeyError if missing
        original_query = item["query"]
        # build_input_images also requires item["image"] and item["image_root"]
        input_images = build_input_images(item)

        if not input_images:
            out = {"id": rid, "error": "no_images", "lang": "en"}
            append_jsonl(out_jsonl, out)
            return out

        rewritten_query = rewrite_query_with_reasoner(original_query)
        img_meta_info = get_image_meta(img_meta_pre, input_images)
        # gt is optional
        gt = item.get("gt", None)

        data_sample = {
            "query": rewritten_query,
            "meta_information": img_meta_info,
            "image": input_images,
            "gt": gt,
        }

        loop_out = socraticLoop(
            r_model, p_model, data_sample,
            max_loop=max_loop,
            verify_model=verify_model,
            verify_inst=verify_inst,
            debug=debug
        )

        out = {
            "id": rid,
            "task": item.get("task"),
            "data_source": item.get("data_source"),
            "query": original_query,
            "rewritten_query": rewritten_query,
            "gt": gt,
            "image": item["image"],          # required in format, so [] won't raise
            "image_root": item["image_root"],
            "loop_result": loop_out,
            "lang": "en",
        }
        append_jsonl(out_jsonl, out)
        return out

    except Exception as e:
        err = {
            "id": rid,
            "error": f"worker_exception:{type(e).__name__}",
            "error_msg": str(e),
            "lang": "en"
        }
        append_jsonl(out_jsonl, err)
        return err


# =========================
# Reporting
# =========================

def read_jsonl(path: str):
    if not os.path.exists(path):
        return []
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    return rows


def build_report(all_count: int, used_count: int, rows_iter) -> Dict[str, Any]:
    success = 0
    failure = 0
    fail_buckets: Dict[str, int] = {}
    seen = set()

    for rec in rows_iter:
        rid = rec.get("id")
        if rid in seen:
            continue
        seen.add(rid)

        lr = rec.get("loop_result")
        if isinstance(lr, dict) and lr.get("success") is True:
            success += 1
        else:
            failure += 1
            reason = None
            if isinstance(lr, dict):
                reason = lr.get("error")
            if reason is None:
                reason = rec.get("error")
            if not reason:
                reason = "unknown_error"
            fail_buckets[reason] = fail_buckets.get(reason, 0) + 1

    sorted_items = sorted(fail_buckets.items(), key=lambda kv: (-kv[1], str(kv[0])))
    return {
        "total_dataset_size": all_count,
        "attempted_items": used_count,
        "processed_unique": len(seen),
        "success": success,
        "failure": failure,
        "failure_breakdown": dict(sorted_items),
        "lang": "en",
        "concurrency": None,  # filled later
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }


# =========================
# CLI
# =========================

def parse_args():
    parser = argparse.ArgumentParser(description="Socratic loop data generation (English only, generic parquet).")
    parser.add_argument("--data-path", type=str, required=True, help="Path to input parquet file.")
    parser.add_argument("--output-dir", type=str, default=None, help="Directory for outputs. Default: same as data-path dir.")
    parser.add_argument("--debug", action="store_true", default=False, help="Enable debug mode.")
    parser.add_argument("--debug-samples", type=int, default=8, help="Number of samples in debug mode.")
    parser.add_argument("--concurrency", type=int, default=15, help="Thread pool concurrency.")
    parser.add_argument("--resume", action="store_true", default=False, help="Enable resume from existing JSONL (ignored in debug).")
    parser.add_argument("--max-loop", type=int, default=8, help="Maximum reasoning loop rounds.")
    parser.add_argument("--current-task-samples", type=int, default=500, help="Max samples to process this run.")
    parser.add_argument("--verify-inst", type=str, default=None, help="Verifier instruction. If omitted, verifier is disabled.")
    parser.add_argument(
        "--image-meta-pre",
        type=str,
        required=True,
        help="Prefix text for image metadata, e.g. 'The provided imagery is remote sensing image"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    DATA_PATH = args.data_path
    DATA_BASENAME = os.path.splitext(os.path.basename(DATA_PATH))[0]

    DEBUG = args.debug
    DEBUG_SAMPLES = args.debug_samples
    CONCURRENCY = args.concurrency
    RESUME = args.resume and (not DEBUG)  # no resume in debug mode
    MAX_LOOP = args.max_loop
    CURRENT_TASK_SAMPLES = args.current_task_samples
    VERIFY_INST = args.verify_inst
    IMG_META_PRE = args.image_meta_pre

    # Output paths
    RUN_TAG = "debug" if DEBUG else "full"

    if args.output_dir is not None:
        OUTPUT_DIR = args.output_dir
    else:
        OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(DATA_PATH)), "demo_output") or "."

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if DEBUG:
        TS = datetime.now().strftime("%Y%m%d%H%M%S")
        OUT_JSONL = os.path.join(
            OUTPUT_DIR, f"{DATA_BASENAME}_{RUN_TAG}_{TS}.jsonl"
        )
    else:
        OUT_JSONL = os.path.join(OUTPUT_DIR, f"{DATA_BASENAME}.jsonl")

    TS_NOW = datetime.now().strftime("%Y%m%d%H%M%S")
    OUT_SNAPSHOT_JSON = os.path.join(OUTPUT_DIR, f"{DATA_BASENAME}_{RUN_TAG}_{TS_NOW}.json")
    REPORT_JSON = os.path.join(OUTPUT_DIR, f"{DATA_BASENAME}_{RUN_TAG}_{TS_NOW}_report.json")

    # Models
    r_system_prompt = build_reasoner_system_prompt(MAX_LOOP)
    r_model = APIModel("gpt-5-mini", r_system_prompt)
    p_model = APIModel("gemini-2.5-flash", P_SYSTEM_PROMPT)

    if VERIFY_INST is None:
        print("Verifier not required (verify_inst is None).")
        verify_model = None
    else:
        verify_model = APIModel("doubao-seed-1-6-thinking-250715", VERIFY_SYSTEM_PROMPT)

    # Data
    # Expected parquet schema per row: see process_item() docstring for details.
    data = datasets.load_dataset("parquet", data_files=DATA_PATH)["train"]
    total = len(data)
    if DEBUG:
        dbg_header("DATA SAMPLE", DEBUG)
        dbg(str(data[0] if total > 0 else None), DEBUG)

    # Load processed ids (resume)
    processed_ids = set()
    if RESUME:
        processed_ids = load_processed_ids_from_jsonl(OUT_JSONL)
        print(f"[RESUME] loaded processed ids from {OUT_JSONL}: {len(processed_ids)}")

    # Sampling
    if DEBUG:
        idxs = random.sample(range(total), min(DEBUG_SAMPLES, total))
        candidates_all = [data[i] for i in idxs]
        candidates = candidates_all
        dbg(f"[SAMPLE] DEBUG sampled {len(candidates)} / dataset size {total}", DEBUG)
    else:
        remaining = [it for it in data if get_item_id(it) not in processed_ids]
        pick_n = min(CURRENT_TASK_SAMPLES, len(remaining))
        candidates = random.sample(remaining, pick_n) if pick_n > 0 else []
        print(f"[SAMPLE] remaining={len(remaining)}, pick={pick_n}, dataset={total}")

    items_with_id = [(it, get_item_id(it)) for it in candidates]
    if RESUME:
        items_with_id = [(it, rid) for it, rid in items_with_id if rid not in processed_ids]

    if DEBUG:
        dbg(f"[RUN] to process: {len(items_with_id)} / sampled: {len(candidates)} / dataset: {total}", DEBUG)
    else:
        print(f"[RUN] to process: {len(items_with_id)}")

    # Concurrent execution
    with ThreadPoolExecutor(max_workers=CONCURRENCY) as ex:
        futures = [
            ex.submit(
                process_item,
                it,
                rid,
                r_model,
                p_model,
                verify_model,
                VERIFY_INST,
                IMG_META_PRE,
                MAX_LOOP,
                OUT_JSONL,
                DEBUG
            )
            for it, rid in items_with_id
        ]
        if not DEBUG:
            for _ in tqdm(as_completed(futures), total=len(futures), desc="Processing", ncols=90):
                pass
        else:
            for fut in tqdm(as_completed(futures), total=len(futures), desc="Processing (DEBUG)", ncols=90):
                _ = fut.result()

    # Build report
    all_rows = read_jsonl(OUT_JSONL)
    used_count = len(items_with_id)
    report = build_report(all_count=total, used_count=used_count, rows_iter=all_rows)
    report["run_tag"] = RUN_TAG
    report["concurrency"] = CONCURRENCY

    # Snapshot and report
    latest_map: Dict[str, Dict[str, Any]] = {}
    for rec in all_rows:
        rid = rec.get("id") or str(uuid.uuid4())
        latest_map[rid] = rec
    snapshot_rows = [latest_map.get(rid) for _, rid in items_with_id if latest_map.get(rid)]
    with open(OUT_SNAPSHOT_JSON, "w", encoding="utf-8") as f:
        json.dump(snapshot_rows, f, ensure_ascii=False, indent=2)
    with open(REPORT_JSON, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("Saved snapshot to:", OUT_SNAPSHOT_JSON)
    print("Saved report   to:", REPORT_JSON)
    print("JSONL (main):  ", OUT_JSONL)
    print("Report Summary:", json.dumps(report, ensure_ascii=False, indent=2))
