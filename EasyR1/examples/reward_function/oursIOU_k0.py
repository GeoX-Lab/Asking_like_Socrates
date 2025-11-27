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

def _unpack_gt_bundle(gt: Any, idx: int) -> Tuple[List[float]]:
    """
    Unpack ground_truth to (bbox,), requiring:
        ground_truth == { "bbox": [x1,y1,x2,y2] in [0,1000] }
    """
    gt = _unwrap_np_object(gt)
    if isinstance(gt, dict):
        if "bbox" not in gt:
            raise ValueError(f"Sample {idx}: ground_truth dict missing key 'bbox'.")
        bbox = _parse_box_any4(gt["bbox"])
        return (bbox,)
    try:
        _ = _parse_box_any4(gt)
    except Exception:
        raise ValueError(f"Sample {idx}: ground_truth must be a dict with bbox, got type {type(gt)}.")
    raise ValueError(f"Sample {idx}: ground_truth lacks required key 'bbox'.")

# -------------------- Response parsing utilities --------------------

_THINK_CLOSE_RE = re.compile(r"</\s*think\s*>", flags=re.IGNORECASE)
ANSWER_RE = re.compile(r"(?i)\banswer\s*[:：]")

def _has_think_close(response: str) -> bool:
    """True if a closing </think> exists (opening may be omitted)."""
    return bool(_THINK_CLOSE_RE.search(response))

def _first_think_close_pos(response: str) -> Optional[int]:
    """End-position of the FIRST </think> closing tag."""
    m = _THINK_CLOSE_RE.search(response)
    return m.end() if m else None

# --- bbox object parsing: {"bbox":[...]} (lenient JSON) ---

def _extract_first_bbox_in(text: str) -> Optional[List[float]]:
    """
    从 text 中提取第一个 {"bbox":[...]}（宽松匹配）并返回前4个、范围在[0,1000]的浮点数。
    允许键名不带引号：{bbox:[...]} 也可。
    """
    m = re.search(r'["\']?\s*bbox\s*["\']?\s*:\s*\[\s*([^\]]+)\]', text, flags=re.IGNORECASE | re.DOTALL)
    if not m:
        return None
    coords_txt = m.group(1)
    nums = re.findall(_NUMBER, coords_txt)
    if len(nums) < 4:
        return None
    vals: List[float] = []
    for s in nums[:4]:
        v = float(s)
        if not (0.0 <= v <= 1000.0):
            return None
        vals.append(v)
    return vals

def _extract_first_any4_bracket(text: str) -> Optional[List[float]]:
    """
    在 text 中寻找第一个方括号 '[ ... ]'，若可解析出4个位于[0,1000]的数字，则返回这4个数字。
    用作无 'bbox:' 键名时的回退解析。
    """
    # 逐个匹配方括号内容，遇到能解析出4个合法数字的就返回
    for m in re.finditer(r'\[\s*([^\]]+)\]', text, flags=re.DOTALL):
        nums = re.findall(_NUMBER, m.group(1))
        if len(nums) < 4:
            continue
        vals: List[float] = []
        ok = True
        for s in nums[:4]:
            v = float(s)
            if not (0.0 <= v <= 1000.0):
                ok = False
                break
            vals.append(v)
        if ok:
            return vals
    return None

def _has_any_bbox_before(text: str, end_pos: int) -> bool:
    """
    是否在 text[:end_pos] 范围内出现过任何 {"bbox":[...]}。
    仅做宽松检测：出现 'bbox:[' 并有至少一个数字就算。
    """
    prefix = text[:max(0, end_pos)]
    return bool(re.search(r"\bbbox\s*:\s*\[\s*" + _NUMBER, prefix, flags=re.IGNORECASE | re.DOTALL))

def _parse_pred_bbox_after_think(response: str) -> Tuple[Optional[List[float]], bool]:
    """
    解析 </think> 之后的预测框：
      1) 若存在 'Answer:'，优先在其后提取 {"bbox":[...]}；
         若失败，再回退到任意 '[...]' 的 4 数字解析（used_fallback=True）。
      2) 若不存在 'Answer:'，直接在 </think> 之后的 tail 依次尝试上面两种方法。
    返回: (bbox_or_None, used_fallback_bool)
    """
    if not _has_think_close(response):
        return None, False
    close_pos = _first_think_close_pos(response)
    if close_pos is None:
        return None, False
    tail = response[close_pos:]

    # 尝试 Answer 之后
    m_ans = ANSWER_RE.search(tail)
    if m_ans:
        after_ans = tail[m_ans.end():]
        bbox = _extract_first_bbox_in(after_ans)
        if bbox is not None:
            return bbox, False
        # 回退：任意 [a,b,c,d]
        any4 = _extract_first_any4_bracket(after_ans)
        if any4 is not None:
            return any4, True
        return None, False

    # 没有 Answer：仍允许解析（准确度可用，但格式分数依然按是否有 Answer 计分）
    bbox = _extract_first_bbox_in(tail)
    if bbox is not None:
        return bbox, False
    any4 = _extract_first_any4_bracket(tail)
    if any4 is not None:
        return any4, True
    return None, False

def _parse_answer_bbox_after_think(response: str) -> Optional[List[float]]:
    """
    兼容旧接口：只返回 bbox（优先 JSON 形式；若无则回退到任意4数方括号）。
    """
    bbox, _ = _parse_pred_bbox_after_think(response)
    return bbox

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
    """IoU in 0..1000 space (result in [0,1])."""
    ax1, ay1, ax2, ay2 = _fix_and_clip_box_xyxy01(a)
    bx1, by1, bx2, by2 = _fix_and_clip_box_xyxy01(b)
    iw = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    ih = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return 0.0 if union <= 0.0 else inter / union

# -------------------- Accuracy (IoU only) --------------------

def accuracy_reward_boxes_with_fallback(
    response: str,
    gt_bbox: Sequence[float],
) -> Tuple[float, float, bool]:
    """
    与原版类似，但返回一个 used_fallback 标志：
      - True: 使用了 </think> 后的任意 '[...]' 4 数字回退解析
      - False: 使用了标准 {'bbox':[...]} 解析
    返回: (accuracy_score, IoU_raw, used_fallback)
    """
    # 轻度归一化尖括号周围空白，避免匹配失败
    response = re.sub(r"\s*(<|>|/)\s*", r"\1", response)

    pred, used_fallback = _parse_pred_bbox_after_think(response)
    if pred is None:
        return 0.0, 0.0, False

    iou = _iou_xyxy01(pred, gt_bbox)
    iou_score = float(max(0.0, min(1.0, iou)))
    return float(iou_score), float(iou), bool(used_fallback)

def accuracy_reward_boxes(
    response: str,
    gt_bbox: Sequence[float],
) -> Tuple[float, float]:
    """
    兼容旧签名：仅返回 (accuracy_score, IoU_raw)。
    """
    acc, iou, _ = accuracy_reward_boxes_with_fallback(response, gt_bbox)
    return acc, iou

# -------------------- Format reward (3 parts + early-bbox penalty) --------------------

def format_reward_detailed(response: str) -> Dict[str, float]:
    """
    格式奖励（只要求 '</think>'；不要求成对 '<think>...</think>'）：
      (1) format_thinkclose  -> 是否存在 '</think>' (0/1)
      (2) format_answer      -> '</think>' 之后是否出现 'Answer:' / 'Answer：' (0/1)
      (3) format_bbox_after  -> 该 'Answer' 之后是否出现合法 {'bbox':[x1,y1,x2,y2]} (0/1)
      format_total = (三项平均)；若任何 {'bbox':[]} 出现在 '</think>' 之前，最终总分减半。

    说明：是否采用“任意 [a,b,c,d] 回退解析”的减半，不在本函数内部处理，
         由外层（compute_score）在 used_fallback=True 时再额外减半。
    """
    score_thinkclose = 0.0
    score_answer = 0.0
    score_bbox_after = 0.0
    penalty_half = False

    if not isinstance(response, str) or not response.strip():
        return {
            "format_total": 0.0,
            "format_thinkclose": 0.0,
            "format_answer": 0.0,
            "format_bbox_after": 0.0,
        }

    # 轻度规范化尖括号空白
    response = re.sub(r"\s*(<|>|/)\s*", r"\1", response)

    # 1) '</think>'
    if _has_think_close(response):
        score_thinkclose = 1.0
        close_pos = _first_think_close_pos(response)
        if close_pos is not None:
            tail = response[close_pos:]
            bbox = _extract_first_bbox_in(tail)
            if bbox is not None:
                score_bbox_after = 1.0
            elif _has_any_bbox_before(response, close_pos):
                    # 惩罚检测：是否有 bbox 出现在 '</think>' 之前（仅针对带 'bbox:' 的JSON样式）
                    penalty_half = True

    format_total = (score_thinkclose  + score_bbox_after) / 2.0
    if penalty_half and format_total > 0.0:
        format_total *= 0.5

    return {
        "format_total": float(format_total),
        "format_thinkclose": float(score_thinkclose),
        "format_answer": float(score_answer),
        "format_bbox_after": float(score_bbox_after),
    }

def format_reward(response: str) -> float:
    """兼容封装：返回格式总分（含“bbox在 think 前”减半惩罚）。"""
    return format_reward_detailed(response)["format_total"]

# -------------------- E requirement --------------------

def _require_shortcut_E(item: Dict[str, Any],
                        validated: Dict[str, Any],
                        sample_idx: int) -> float:
    """
    Require shortcut expectation E in [0,1].
    Allowed sources:
      - item['answer']['E']
      - item['ground_truth']['E']
      - validated.get('E')
    """
    candidates = [
        ("item['answer']['E']", (item.get("answer") or {}).get("E", None)),
        ("item['ground_truth']['E']", (item.get("ground_truth") or {}).get("E", None)),
        ("validated['E']", validated.get("E", None)),
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
        "Expected at one of: item['answer']['E'], item['ground_truth']['E'], validated['E']."
    )

# -------------------- Dataset validation --------------------

def _validate_dataset_item(item: Dict[str, Any], idx: int) -> Dict[str, Any]:
    """
    STRICT validation (simplified):
      - ground_truth: {'bbox': [x1,y1,x2,y2] in [0,1000]}
    Returns: {'gt_bbox'}
    """
    if "ground_truth" not in item:
        raise ValueError(f"Sample {idx}: missing 'ground_truth'")
    (gt_bbox,) = _unpack_gt_bundle(item["ground_truth"], idx)
    return {"gt_bbox": gt_bbox}

# -------------------- Batch scoring --------------------

def compute_score(
    reward_inputs: List[Dict[str, Any]],
    format_weight: float = 0.1,
) -> List[Dict[str, float]]:
    """
    overall = (1 - format_weight) * ((accuracy - E) / (1 - E))
              + format_weight * format_total

    - accuracy: IoU-only (pred bbox = 'Answer:' 后的 {"bbox":[...]}；若无则回退到任意 '[a,b,c,d]' 解析)
    - format_total: 三项平均；若有早产 bbox（出现在 </think> 前），再把总分减半
    - 新增：若采用了“任意 '[a,b,c,d]' 回退解析”，则在上述基础上再把 format_total 减半
    """
    if not isinstance(reward_inputs, list):
        raise ValueError("Please use `reward_type=batch` for boxes reward function (reward_inputs must be a list).")

    fw = float(format_weight)
    if not (0.0 <= fw <= 1.0):
        raise ValueError(f"format_weight out of [0,1]: {format_weight}")

    results: List[Dict[str, float]] = []
    for idx, item in enumerate(reward_inputs):
        v = _validate_dataset_item(item, idx)

        response = str(item.get("response", ""))

        # 1) accuracy (IoU-only) + 是否使用回退解析
        acc_raw, iou, used_fallback = accuracy_reward_boxes_with_fallback(response, v["gt_bbox"])

        # 2) detailed format rewards
        fmt_detail = format_reward_detailed(response)
        fmt_total = fmt_detail["format_total"]

        # 3) 若使用了“任意 [a,b,c,d] 回退解析”，格式分数再减半
        if used_fallback and fmt_total > 0.0:
            fmt_total *= 0.5

        # 4) E (shortcut expectation)
        E = _require_shortcut_E(item, v, idx)
        #print(E)
        denom = (1.0 - E)
        if denom <= 1e-12:
            acc_adj = 1.0 if acc_raw >= 1.0 - 1e-12 else 0.0
        else:
            acc_adj = (acc_raw - E) / denom
        #print(acc_adj)
        base_w = 1.0 - fw
        overall = base_w * acc_adj + fw * fmt_total

        results.append({
            "overall": overall,
            "accuracy_raw": float(acc_raw),
            "accuracy": acc_adj,
            "IOU": float(iou),
            "format_total": float(fmt_total),
            "format_thinkclose": float(fmt_detail["format_thinkclose"]),
            "format_answer": float(fmt_detail["format_answer"]),
            "format_bbox_after": float(fmt_detail["format_bbox_after"]),
            # 可选：暴露一个标志，方便下游统计
            "used_fallback_any4": float(1.0 if used_fallback else 0.0),
        })
    return results
