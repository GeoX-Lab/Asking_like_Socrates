import re
import math
from typing import Any, Optional, Sequence, Tuple, List, Dict

_NUMBER = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"

# -------------------- Parsing helpers --------------------

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
    Parse 4 floats in [0,1000] from list/tuple/numpy (not from string).
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
    if not all(0.0 <= float(x) <= 1000.0 for x in vals):
        raise ValueError(f"BBox values out of [0,1000]: {vals}")
    return [float(x) for x in vals]

# -------------------- Response parsing: think block + JSON {"bbox":[...]} --------------------

_THINK_BLOCK_RE = re.compile(r"<\s*think\s*>.*?</\s*think\s*>", flags=re.IGNORECASE | re.DOTALL)
_THINK_CLOSE_RE = re.compile(r"</\s*think\s*>", flags=re.IGNORECASE)

# JSON bbox:  Answer: {"bbox": [x1, y1, x2, y2]}  或 任意位置出现 {"bbox":[...]}
_JSON_BBOX_RE = re.compile(
    r'"\s*bbox\s*"\s*:\s*\[\s*(' + _NUMBER + r')\s*,\s*(' + _NUMBER + r')\s*,\s*(' + _NUMBER + r')\s*,\s*(' + _NUMBER + r')\s*\]',
    flags=re.IGNORECASE | re.DOTALL
)

def _has_think_block(response: str) -> bool:
    """Return True if a paired <think>...</think> exists."""
    return bool(_THINK_BLOCK_RE.search(response))

def _first_think_close_pos(response: str) -> Optional[int]:
    """Return end-position of the FIRST </think> closing tag."""
    m = _THINK_CLOSE_RE.search(response)
    return m.end() if m else None

def _parse_json_bbox(text: str) -> Optional[List[float]]:
    """Parse FIRST {"bbox":[x1,y1,x2,y2]} with all four numbers in [0,1000]."""
    m = _JSON_BBOX_RE.search(text)
    if not m:
        return None
    vals = [float(m.group(i)) for i in range(1, 5)]
    # order & clip
    x1, y1, x2, y2 = vals
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    x1 = max(0.0, min(1000.0, x1))
    y1 = max(0.0, min(1000.0, y1))
    x2 = max(0.0, min(1000.0, x2))
    y2 = max(0.0, min(1000.0, y2))
    return [x1, y1, x2, y2]

def _parse_bbox_after_think(response: str) -> Optional[List[float]]:
    """
    Enforce ORDERED placement:
      - <think>...</think> must exist
      - AFTER the FIRST </think>, there must be a JSON {"bbox":[x1,y1,x2,y2]} in [0,1000]
    Return parsed [x1,y1,x2,y2] if all satisfied, else None.
    """
    if not _has_think_block(response):
        return None
    close_pos = _first_think_close_pos(response)
    if close_pos is None:
        return None
    tail = response[close_pos:]
    return _parse_json_bbox(tail)

def _bbox_presence_regions(response: str) -> Dict[str, bool]:
    """
    返回是否在 </think> 前/后找到合法的 {"bbox":[...]}。
    若不存在 <think>...</think>，两者均为 False。
    """
    has_think = _has_think_block(response)
    if not has_think:
        return {"before": False, "after": False}
    close_pos = _first_think_close_pos(response)
    if close_pos is None:
        return {"before": False, "after": False}
    head = response[:close_pos]
    tail = response[close_pos:]
    return {
        "before": _parse_json_bbox(head) is not None,
        "after": _parse_json_bbox(tail) is not None,
    }

# -------------------- Geometry & IoU in 0..1000 space --------------------

def _fix_and_clip_box_xyxy01(box: Sequence[float]) -> Tuple[float, float, float, float]:
    """Ensure (x1<=x2, y1<=y2) and clip to [0,1000]."""
    x1, y1, x2, y2 = box
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    x1 = max(0.0, min(1000.0, x1))
    y1 = max(0.0, min(1000.0, y1))
    x2 = max(0.0, min(1000.0, x2))
    y2 = max(0.0, min(1000.0, y2))
    return x1, y1, x2, y2

def _iou_xyxy01(a: Sequence[float], b: Sequence[float]) -> float:
    """IoU in 0..1000 space (result ∈ [0,1])."""
    ax1, ay1, ax2, ay2 = _fix_and_clip_box_xyxy01(a)
    bx1, by1, bx2, by2 = _fix_and_clip_box_xyxy01(b)
    iw = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    ih = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return 0.0 if union <= 0.0 else inter / union

# -------------------- Accuracy (IoU only; JSON-after-think) --------------------

def accuracy_reward_boxes(
    response: str,
    gt_bbox: Sequence[float],
) -> Tuple[float, float]:
    """
    Accuracy using ONLY IoU:
        IoU_score = exp((IoU*2.5) - 1) / (e**2.5 - 1)
        acc       = IoU_score

    - Pred bbox: parsed from JSON {"bbox":[...]} that appears AFTER the first </think>.
      If any condition fails, return (0.0, 0.0).
    """
    # normalize angle bracket spacing to avoid matching issues
    response = re.sub(r"\s*(<|>|/)\s*", r"\1", str(response))

    pred = _parse_bbox_after_think(response)
    if pred is None:
        return 0.0, 0.0

    iou = _iou_xyxy01(pred, gt_bbox)

    # IoU -> score mapping (smooth, convex-up)
    denom = math.exp(2.5) - 1.0
    iou_score = (math.exp((iou * 2.5) - 1.0) / denom) if denom > 0 else 0.0
    iou_score = float(max(0.0, min(1.0, iou_score)))

    return float(iou_score), float(iou)

# -------------------- Dataset validation (ground_truth: tag/bbox/E) --------------------

def _validate_dataset_item(item: Dict[str, Any], idx: int) -> Dict[str, Any]:
    """
    STRICT validation:
      item must contain:
        - "ground_truth": { "tag": "iou", "bbox": [x1,y1,x2,y2] in [0,1000], "E": float in [0,1] }
        - "response": str (may be empty)
    Returns: {"gt_bbox": List[float], "E": float}
    """
    if "ground_truth" not in item:
        raise ValueError(f"Sample {idx}: missing 'ground_truth'")

    gt = _unwrap_np_object(item["ground_truth"])
    if not isinstance(gt, dict):
        raise ValueError(f"Sample {idx}: ground_truth must be a dict, got {type(gt)}")

    tag = gt.get("tag", None)
    if tag is None:
        raise ValueError(f"Sample {idx}: ground_truth missing 'tag'")
    if str(tag).lower() != "iou":
        raise ValueError(f"Sample {idx}: unsupported ground_truth.tag='{tag}', expected 'iou'.")

    if "bbox" not in gt:
        raise ValueError(f"Sample {idx}: ground_truth dict missing key 'bbox'.")
    gt_bbox = _parse_box_any4(gt["bbox"])

    if "E" not in gt:
        raise ValueError(f"Sample {idx}: ground_truth dict missing key 'E'.")
    try:
        E = float(gt["E"])
    except Exception:
        raise ValueError(f"Sample {idx}: ground_truth['E'] is not a valid float (got: {gt['E']!r}).")
    if not (0.0 <= E <= 1.0):
        raise ValueError(f"Sample {idx}: E must be in [0,1], got {E}.")

    _ = str(item.get("response", ""))  # ensure exists for downstream parsing

    return {"gt_bbox": gt_bbox, "E": E}

# -------------------- Format reward (JSON-after-think + half-penalty rule) --------------------

def format_reward_detailed(response: str) -> Dict[str, float]:
    """
    细粒度格式奖励：
      (1) format_think  -> <think>...</think> 成对出现 (0/1)
      (2) format_answer -> 在 </think> 后出现 'Answer:' 或 'Answer：' (0/1)  [可选项，兼容旧提示]
      (3) format_box    -> 在 </think> 后存在且可解析的 {"bbox":[x1,y1,x2,y2]} (0/1)

      基础分：format_total = (format_think + format_answer + format_box) / 3

      额外条款（你要求的）：
      - 如果 {"bbox":...} 不存在于 </think> 后面，
        但在 </think> 前面能找到合法的 {"bbox":...}，
        则对 format_total 施加 0.5 倍惩罚（减半）。
    """
    if not isinstance(response, str) or not response.strip():
        return {
            "format_total": 0.0,
            "format_think": 0.0,
            "format_answer": 0.0,
            "format_box": 0.0,
        }

    # 规范尖括号附近空白，避免奇怪写法影响匹配
    response = re.sub(r"\s*(<|>|/)\s*", r"\1", response)

    score_think = 1.0 if _has_think_block(response) else 0.0

    # 判定 Answer: 是否出现在 </think> 之后（兼容旧提示，非必须）
    score_answer = 0.0
    close_pos = _first_think_close_pos(response) if score_think else None
    if close_pos is not None:
        tail = response[close_pos:]
        if re.search(r"(?i)\banswer\s*[:：]", tail):
            score_answer = 1.0

    # JSON bbox 出现在 </think> 之后？
    score_box = 0.0
    if close_pos is not None:
        tail = response[close_pos:]
        if _parse_json_bbox(tail) is not None:
            score_box = 1.0

    format_total = (score_think + score_answer + score_box) / 3.0

    # 额外条款：仅在 </think> 前存在 bbox，而 </think> 后不存在 -> 减半
    pres = _bbox_presence_regions(response) if score_think else {"before": False, "after": False}
    if score_think == 1.0 and pres["after"] is False and pres["before"] is True:
        format_total *= 0.5  # 惩罚系数

    return {
        "format_total": float(format_total),
        "format_think": float(score_think),
        "format_answer": float(score_answer),
        "format_box": float(score_box),
    }

def format_reward(response: str) -> float:
    """兼容旧接口：返回格式总分（0~1）。"""
    return format_reward_detailed(response)["format_total"]

# -------------------- Batch scoring (E only from ground_truth) --------------------

def compute_score(
    reward_inputs: List[Dict[str, Any]],
    format_weight: float = 0.1,   # weight for the overall format reward (averaged)
) -> List[Dict[str, float]]:
    """
    overall = (1 - format_weight) * ((accuracy_raw - E) / (1 - E))
              + format_weight * format_total

    - accuracy_raw: IoU-only mapping from accuracy_reward_boxes()
    - format_total: (format_think + format_answer + format_box)/3, with the half-penalty rule
    - E: MUST come from item['ground_truth']['E'] in [0,1]
    """
    if not isinstance(reward_inputs, list):
        raise ValueError("Please use `reward_type=batch` for boxes reward function (reward_inputs must be a list).")

    fw = float(format_weight)
    if not (0.0 <= fw <= 1.0):
        raise ValueError(f"format_weight out of [0,1]: {format_weight}")

    results: List[Dict[str, float]] = []
    for idx, item in enumerate(reward_inputs):
        v = _validate_dataset_item(item, idx)

        response = re.sub(r"\s*(<|>|/)\s*", r"\1", str(item.get("response", "")))

        # accuracy (IoU-only, requires JSON {"bbox":[...]} after first </think>)
        acc_raw, iou = accuracy_reward_boxes(response, v["gt_bbox"])

        # detailed format rewards (with the new penalty)
        fmt_detail = format_reward_detailed(response)
        fmt_total = fmt_detail["format_total"]

        # E from ground_truth only
        E = v["E"]
        acc_adj = (acc_raw - E) / (1 - E) if E < 1.0 else 0.0

        base_w = 1.0 - fw
        overall = base_w * acc_adj + fw * fmt_total

        results.append({
            "overall": float(overall),
            "accuracy_raw": float(acc_raw),
            "accuracy": float(acc_adj),
            "IOU": float(iou),
            "format_total": float(fmt_total),
            "format_think": float(fmt_detail["format_think"]),
            "format_answer": float(fmt_detail["format_answer"]),
            "format_box": float(fmt_detail["format_box"]),
        })
    return results
