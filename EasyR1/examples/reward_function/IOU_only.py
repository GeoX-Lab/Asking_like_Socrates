# Reward for normalized bounding boxes (must be inside \boxed{...})
# IoU-based accuracy: R_iou = expm1(IoU) / expm1(1)
# New combined accuracy:
#   acc = (1 - center_weight) * iou_score + center_weight * center_score
# where:
#   D = Euclidean distance between centers in [0,1]^2
#   r = 0.5 * diagonal_length(gt_box)  # adaptive radius
#   center_score = r / max(D, r)       # in (0, 1], saturates at 1 when D <= r

import re
import math
from typing import Any, Optional, Sequence, Tuple, List, Dict

_NUMBER = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"

# -------------------- Parsing helpers --------------------

def _extract_boxed(text: str) -> Optional[str]:
    """Return content inside \boxed{...}; None if absent."""
    m = re.search(r"\\boxed\{\s*(.*?)\s*\}", text, flags=re.DOTALL)
    return m.group(1) if m else None

def _extract_first_think(text: str) -> Optional[str]:
    """Return content inside the FIRST <think>...</think>; None if absent."""
    m = re.search(r"<think>\s*(.*?)\s*</think>", text, flags=re.DOTALL | re.IGNORECASE)
    return m.group(1) if m else None

def _parse_four_floats_0_1(text: str) -> Optional[List[float]]:
    """Parse the FIRST four floats in [0,1] from text."""
    nums = re.findall(_NUMBER, text)
    if len(nums) < 4:
        return None
    vals: List[float] = []
    for s in nums[:4]:
        try:
            v = float(s)
        except Exception:
            return None
        if not (0.0 <= v <= 1.0):
            return None
        vals.append(v)
    return vals if len(vals) == 4 else None

def _parse_box_from_response_boxed_only(response: str) -> Optional[List[float]]:
    """Must parse from inside \boxed{...}; otherwise None."""
    inner = _extract_boxed(response)
    if inner is None:
        return None
    return _parse_four_floats_0_1(inner)

# -------------------- GT bundle parsing --------------------

def _unwrap_np_object(x: Any) -> Any:
    """If x is a 0-d numpy object array, return its Python object."""
    try:
        import numpy as np
        if isinstance(x, np.ndarray) and x.dtype == object and x.ndim == 0:
            return x.item()
    except Exception:
        pass
    return x

def _parse_box_any4(v: Any) -> List[float]:
    """
    Parse 4 floats in [0,1] from list/tuple/numpy (not from string).
    Raise on failure.
    """
    try:
        import numpy as np
        if isinstance(v, np.ndarray):
            arr = v.astype(float).reshape(-1)
            if arr.size < 4:
                raise ValueError(f"GT bbox ndarray must have >=4 elements, got {arr.size}")
            vals = arr[:4].tolist()
        elif isinstance(v, (list, tuple)):
            if len(v) != 4:
                raise ValueError(f"GT bbox must have 4 elements, got {len(v)}: {v}")
            vals = [float(x) for x in v]
        else:
            raise TypeError(f"Unsupported bbox container type: {type(v)}")
    except Exception as e:
        raise ValueError(f"Failed to parse bbox: {e}")
    if not all((isinstance(x, float) or isinstance(x, int)) for x in vals):
        raise ValueError(f"BBox values are not numeric: {vals}")
    if not all(0.0 <= float(x) <= 1.0 for x in vals):
        raise ValueError(f"BBox values out of [0,1]: {vals}")
    return [float(x) for x in vals]

def _unpack_gt_bundle(gt: Any, idx: int) -> Tuple[List[float], int, int, str]:
    """
    Unpack ground_truth to (bbox, think_words, wait_repeats, think_opening).
    Expected schema (new):
        ground_truth == {
            "bbox": [x1,y1,x2,y2] in [0,1],
            "think_words": int>=1,
            "wait_repeats": int>=0,
            "think_opening": str non-empty
        }
    Backward-compat:
        - If ground_truth is list/tuple/ndarray of 4 floats => raise (missing controls).
        - If ground_truth is numpy 0-d object array wrapping a dict => unwrap and parse.
    """
    gt = _unwrap_np_object(gt)

    # Primary: dict schema
    if isinstance(gt, dict):
        if "bbox" not in gt:
            raise ValueError(f"Sample {idx}: ground_truth dict missing key 'bbox'.")
        bbox = _parse_box_any4(gt["bbox"])

        # controls
        if "think_words" not in gt or not isinstance(gt["think_words"], int) or gt["think_words"] < 1:
            raise ValueError(f"Sample {idx}: ground_truth invalid 'think_words' ({gt.get('think_words')}).")
        if "wait_repeats" not in gt or not isinstance(gt["wait_repeats"], int) or gt["wait_repeats"] < 0:
            raise ValueError(f"Sample {idx}: ground_truth invalid 'wait_repeats' ({gt.get('wait_repeats')}).")
        if "think_opening" not in gt or not isinstance(gt["think_opening"], str) or not gt["think_opening"].strip():
            raise ValueError(f"Sample {idx}: ground_truth invalid 'think_opening' ({gt.get('think_opening')}).")
        return bbox, int(gt["think_words"]), int(gt["wait_repeats"]), gt["think_opening"].strip()

    # Backward-compat (old dataset): plain 4 numbers -> we now consider **invalid**
    # because the new reward requires controls from ground_truth.
    try:
        bbox_only = _parse_box_any4(gt)
    except Exception:
        # Not parsable -> raise a uniform error
        raise ValueError(f"Sample {idx}: ground_truth must be a dict with bbox+controls, got type {type(gt)}.")
    # If it reaches here, bbox was parsable but controls missing:
    raise ValueError(
        f"Sample {idx}: ground_truth lacks thinking controls "
        f"(expect dict with keys: bbox/think_words/wait_repeats/think_opening)."
    )

# -------------------- Geometry & IoU --------------------

def _fix_and_clip_box_xyxy01(box: Sequence[float]) -> Tuple[float, float, float, float]:
    """Ensure (x1<=x2, y1<=y2) and clip to [0,1]."""
    x1, y1, x2, y2 = box
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    x1 = max(0.0, min(1.0, x1))
    y1 = max(0.0, min(1.0, y1))
    x2 = max(0.0, min(1.0, x2))
    y2 = max(0.0, min(1.0, y2))
    return x1, y1, x2, y2

def _iou_xyxy01(a: Sequence[float], b: Sequence[float]) -> float:
    ax1, ay1, ax2, ay2 = _fix_and_clip_box_xyxy01(a)
    bx1, by1, bx2, by2 = _fix_and_clip_box_xyxy01(b)
    iw = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    ih = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return 0.0 if union <= 0.0 else inter / union

def _center_distance_xyxy01(a: Sequence[float], b: Sequence[float]) -> float:
    """Euclidean distance between centers of two clipped boxes in [0,1]^2."""
    ax1, ay1, ax2, ay2 = _fix_and_clip_box_xyxy01(a)
    bx1, by1, bx2, by2 = _fix_and_clip_box_xyxy01(b)
    acx = 0.5 * (ax1 + ax2); acy = 0.5 * (ay1 + ay2)
    bcx = 0.5 * (bx1 + bx2); bcy = 0.5 * (by1 + by2)
    return math.hypot(acx - bcx, acy - bcy)

def _half_diagonal_length_xyxy01(box: Sequence[float]) -> float:
    """Half of the diagonal length of a clipped GT box (in normalized units)."""
    x1, y1, x2, y2 = _fix_and_clip_box_xyxy01(box)
    diag = math.hypot(x2 - x1, y2 - y1)
    return 0.5 * diag

# -------------------- Format & Thinking checks --------------------

def _has_think_block(response: str) -> bool:
    return bool(re.search(r"<think>.*?</think>", response, flags=re.DOTALL | re.IGNORECASE))

def _has_valid_boxed_four(response: str) -> bool:
    boxed_inner = _extract_boxed(response)
    return boxed_inner is not None and (_parse_four_floats_0_1(boxed_inner) is not None)

def _word_count_en(s: str) -> int:
    tokens = re.findall(r"[A-Za-z']+|[0-9]+", s)
    return len(tokens)

def _count_wait_tokens(s: str) -> int:
    return len(re.findall(r"\bwait\b", s, flags=re.IGNORECASE))

def _normalize_spaces(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()

def _strip_quotes(s: str) -> str:
    return s.strip().strip("'\"“”‘’").strip()

def format_reward(
    response: str,
    expected_words: Optional[int],
    expected_waits: Optional[int],
    expected_opening: Optional[str],
) -> Tuple[float, Dict[str, int]]:
    """
    New Format reward: count completed instructions and map via (e^m - 1)/(e^n - 1).
    Instructions considered (binary completion):
      1) Presence of <think>...</think>
      2) Presence of \\boxed{...} with 4 floats in [0,1]
      3) <think> opening matches expected_opening (case-insensitive, prefix match)
      4) <think> word budget respected: words <= expected_words
      5) <think> 'wait' count equals expected_waits (exact match)

    Returns:
      fmt_raw in [0,1], and a dict with counts for debugging: {'m': m, 'n': n}
    Note:
      In compute_score(), this raw score will be multiplied by format_weight,
      i.e., overall adds:  W_format * fmt_raw == W_format * (e^m - 1)/(e^n - 1),
      matching your R_Format definition with W_Format = format_weight.
    """
    # Collect instruction outcomes (True/False)
    instr_flags: List[bool] = []

    # (1) think block present
    has_think = _has_think_block(response)
    instr_flags.append(has_think)

    # (2) boxed valid with 4 normalized floats
    instr_flags.append(_has_valid_boxed_four(response))

    # Prepare think text for the remaining checks
    think_text = _extract_first_think(response) if has_think else None
    t = _normalize_spaces(think_text) if think_text is not None else ""

    # (3) opening matches expected_opening
    if expected_opening and think_text is not None:
        opening_norm = _normalize_spaces(_strip_quotes(expected_opening)).lower()
        opening_ok = t.lower().startswith(opening_norm)
        instr_flags.append(opening_ok)
    else:
        # still count as an instruction if you want n fixed=5; here we only
        # include it when we have an expectation. Given your schema is strict,
        # expectations are always present, so we include it.
        instr_flags.append(False)

    # (4) word budget respected: words <= expected_words
    if isinstance(expected_words, int) and expected_words > 0 and think_text is not None:
        words = _word_count_en(t)
        word_budget_ok = (words <= expected_words)
        instr_flags.append(word_budget_ok)
    else:
        instr_flags.append(False)

    # (5) wait repeats equals expected_waits
    if isinstance(expected_waits, int) and expected_waits >= 0 and think_text is not None:
        wait_count = _count_wait_tokens(t)
        wait_ok = (wait_count == expected_waits)
        instr_flags.append(wait_ok)
    else:
        instr_flags.append(False)

    # m = completed instructions, n = total considered instructions
    # （这里 n 固定为 5；如果未来某些期望缺失，也会以 False 计入，保持与你“之前那几条指令+两结构指令”的定义一致）
    n = len(instr_flags)
    m = sum(1 for f in instr_flags if f)

    # Map via (e^m - 1) / (e^n - 1)
    denom = math.expm1(float(n))
    fmt_raw = (math.expm1(float(m)) / denom) if denom > 0 else 0.0
    fmt_raw = float(max(0.0, min(1.0, fmt_raw)))

    return fmt_raw, {"m": int(m), "n": int(n)}

def thinking_reward(
    response: str,
    expected_words: Optional[int],
    expected_waits: Optional[int],
    expected_opening: Optional[str],
) -> Tuple[float, Dict[str, float]]:
    """
    （保持原来的细粒度思维奖励，不变）
    """
    think_text = _extract_first_think(response)
    if think_text is None:
        return 0.0, {"opening": 0.0, "word_budget": 0.0, "wait": 0.0}

    if expected_words is None and expected_waits is None and not expected_opening:
        return 1.0, {"opening": 0.0, "word_budget": 0.0, "wait": 0.0}

    t = _normalize_spaces(think_text)
    words = _word_count_en(t)
    wait_count = _count_wait_tokens(t)

    opening_score = 1.0
    if expected_opening:
        opening_norm = _normalize_spaces(_strip_quotes(expected_opening)).lower()
        opening_score = 1.0 if t.lower().startswith(opening_norm) else 0.0

    word_budget_score = 1.0
    if isinstance(expected_words, int) and expected_words > 0:
        word_budget_score = 1.0 if words <= expected_words else max(0.0, min(1.0, expected_words / float(max(words, 1))))

    wait_score = 1.0
    if isinstance(expected_waits, int) and expected_waits >= 0:
        if expected_waits == 0:
            wait_score = 1.0 if wait_count == 0 else 0.0
        else:
            diff = abs(wait_count - expected_waits)
            wait_score = max(0.0, 1.0 - diff / float(max(expected_waits, 1)))

    thinking = 0.34 * opening_score + 0.33 * word_budget_score + 0.33 * wait_score
    return float(max(0.0, min(1.0, thinking))), {
        "opening": opening_score,
        "word_budget": word_budget_score,
        "wait": wait_score,
    }

# -------------------- Accuracy (IoU + center) --------------------

def accuracy_reward_boxes(
    response: str,
    gt_bbox: Sequence[float],
    center_weight: float = 0.2,
) -> Tuple[float, float]:
    """
    Combined IoU + center-distance accuracy (read pred ONLY from \\boxed{...}):
        IoU_score    = expm1(IoU(pred, gt)) / expm1(1)
        D            = Euclidean distance between centers (normalized [0,1])
        r            = 0.5 * diagonal_length(gt)
        center_score = r / max(D, r)
        acc          = (1 - center_weight) * IoU_score + center_weight * center_score
    """
    pred = _parse_box_from_response_boxed_only(response)
    if pred is None:
        return 0.0, 0.0

    iou = _iou_xyxy01(pred, gt_bbox)
    denom = math.expm1(1.0)  # e - 1
    iou_score = (math.expm1(iou) / denom) if denom > 0 else 0.0

    D = _center_distance_xyxy01(pred, gt_bbox)
    r = _half_diagonal_length_xyxy01(gt_bbox)
    if r <= 1e-8:
        r = 1e-8
    center_score = r / max(D, r)
    center_score = float(max(0.0, min(1.0, center_score)))

    center_weight = float(max(0.0, min(1.0, center_weight)))
    iou_weight = 1.0 - center_weight
    acc = iou_weight * iou_score + center_weight * center_score
    return float(max(0.0, min(1.0, acc))), float(iou)

# -------------------- Strict dataset validation --------------------

def _validate_dataset_item(item: Dict[str, Any], idx: int) -> Dict[str, Any]:
    """
    STRICT validation for dataset inputs (NEW schema).
    Requires item['ground_truth'] be a dict with keys:
      - 'bbox': list/tuple/np.ndarray of 4 floats in [0,1]
      - 'think_words': int >= 1
      - 'wait_repeats': int >= 0
      - 'think_opening': non-empty str
    Returns: {'gt_bbox', 'tw', 'wr', 'op'}
    """
    if "ground_truth" not in item:
        raise ValueError(f"Sample {idx}: missing 'ground_truth'")

    gt_bbox, tw, wr, op = _unpack_gt_bundle(item["ground_truth"], idx)
    return {"gt_bbox": gt_bbox, "tw": tw, "wr": wr, "op": op}

def _require_shortcut_E(item: Dict[str, Any],
                        validated: Dict[str, Any],
                        sample_idx: int) -> float:
    """
    强制获取样本的捷径期望 E。
    允许的来源（按优先级）：
      1) validated.get("E")        # 若你的 _validate_dataset_item 已解析进来了
      2) item["answer"]["E"]
      3) item["ground_truth"]["E"]
    若以上都不存在或无法转换为 float，抛出 ValueError。
    同时校验 E ∈ [0,1]，否则抛出 ValueError。
    """
    candidates = [
        ("validated['E']", validated.get("E", None)),
        ("item['answer']['E']", (item.get("answer") or {}).get("E", None)),
        ("item['ground_truth']['E']", (item.get("ground_truth") or {}).get("E", None)),
    ]
    for src, v in candidates:
        if v is not None:
            try:
                E = float(v)
            except Exception:
                raise ValueError(f"Sample #{sample_idx}: E from {src} is not a valid float (got: {v!r}).")
            if not (0.0 <= E <= 1.0):
                raise ValueError(f"Sample #{sample_idx}: E must be in [0,1], got {E}.")
            return E
    raise ValueError(
        f"Sample #{sample_idx}: Missing shortcut expectation 'E'. "
        "Expected at one of: validated['E'], item['answer']['E'], item['ground_truth']['E']."
    )


def compute_score(
    reward_inputs: List[Dict[str, Any]],
    format_weight: float = 0.1,   # W_Format
    center_weight: float = 0,   # passed to accuracy_reward_boxes
) -> List[Dict[str, float]]:
    """
    overall = (1 - format_weight) * clip01(accuracy - E)
              + format_weight * fmt_raw
    其中：
      - E 为样本必须提供的捷径期望；若缺失或非法，抛出 ValueError；
      - clip01(x) = min(max(x, 0), 1)；
      - thinking 分数继续计算并返回，但不参与 overall。
    """
    if not isinstance(reward_inputs, list):
        raise ValueError("Please use `reward_type=batch` for boxes reward function (reward_inputs must be a list).")

    fw = float(format_weight)
    if not (0.0 <= fw <= 1.0):
        raise ValueError(f"format_weight out of [0,1]: {format_weight}")

    results: List[Dict[str, float]] = []
    for idx, item in enumerate(reward_inputs):
        # ---------- STRICT dataset validation ----------
        v = _validate_dataset_item(item, idx)
        tw, wr, op = v["tw"], v["wr"], v["op"]

        # ---------- Model output handling (tolerant) ----------
        response = re.sub(r"\s*(<|>|/)\s*", r"\1", str(item.get("response", "")))

        # format
        fmt_raw, fmt_counts = format_reward(response, tw, wr, op)

        # accuracy (returns 0 if pred missing)
        acc, iou = accuracy_reward_boxes(response, v["gt_bbox"], center_weight=center_weight)

        # thinking control reward (kept; not used in overall)
        thinking, think_parts = thinking_reward(response, tw, wr, op)

        # ---------- require shortcut expectation E ----------
        E = _require_shortcut_E(item, v, idx)
        acc_adj = (acc - E)/(1-E)
        
        acc_adj = iou #使用纯IOU进行训练
        
        base_w = 1.0 - fw
        overall = base_w * acc_adj + fw * fmt_raw

        results.append({
            "overall": overall,
            "accuracy_raw": acc,   # 未减 E
            "accuracy": acc_adj,   # 减 E 后并裁剪
            "format": fmt_raw,
            "thinking": thinking,
            "IOU": iou,
            "format_m": float(fmt_counts["m"]),
            "format_n": float(fmt_counts["n"]),
            "thinking_opening": think_parts["opening"],
            "thinking_word_budget": think_parts["word_budget"],
            "thinking_wait": think_parts["wait"],
        })
    return results

