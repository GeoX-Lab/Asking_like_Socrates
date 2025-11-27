import re
import math
from typing import Any, Optional, Sequence, Tuple, List, Dict

_NUMBER = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"

# -------------------- Parsing helpers --------------------

def _extract_boxed(text: str) -> Optional[str]:
    """Return content inside \\boxed{...}; None if absent."""
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

# -------------------- NEW: Parsing custom boxed format --------------------

def _parse_boxed_from_start_end(response: str) -> Optional[List[float]]:
    """Parse coordinates wrapped in <|box_start|> and <|box_end|>."""
    match = re.search(r"<\|box_start\|>\s*(.*?)\s*<\|box_end\|>", response)
    if match:
        coords_str = match.group(1)
        coords = coords_str.split(',')
        try:
            # Convert coordinates to floats and return them
            return [float(coord.strip()) for coord in coords]
        except ValueError:
            return None
    return None

# -------------------- New helper to remove content before </think> --------------------

def _remove_content_before_think(response: str) -> str:
    """
    Remove all content before the first </think> in the response.
    Everything before the first </think> tag is discarded.
    """
    think_close_index = response.lower().find("</think>")
    if think_close_index != -1:
        # Return everything after the </think> tag
        return response[think_close_index + len("</think>"):].strip()
    return response  # If no </think> found, return the original response

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
    Expected schema:
        ground_truth == {
            "bbox": [x1,y1,x2,y2] in [0,1],
            "think_words": int>=1
            "wait_repeats": int>=0
            "think_opening": str non-empty
        }
    """
    gt = _unwrap_np_object(gt)

    if isinstance(gt, dict):
        if "bbox" not in gt:
            raise ValueError(f"Sample {idx}: ground_truth dict missing key 'bbox'.")
        bbox = _parse_box_any4(gt["bbox"])

        if "think_words" not in gt or not isinstance(gt["think_words"], int) or gt["think_words"] < 1:
            raise ValueError(f"Sample {idx}: ground_truth invalid 'think_words' ({gt.get('think_words')}).")
        if "wait_repeats" not in gt or not isinstance(gt["wait_repeats"], int) or gt["wait_repeats"] < 0:
            raise ValueError(f"Sample {idx}: ground_truth invalid 'wait_repeats' ({gt.get('wait_repeats')}).")
        if "think_opening" not in gt or not isinstance(gt["think_opening"], str) or not gt["think_opening"].strip():
            raise ValueError(f"Sample {idx}: ground_truth invalid 'think_opening' ({gt.get('think_opening')}).")
        return bbox, int(gt["think_words"]), int(gt["wait_repeats"]), gt["think_opening"].strip()

    try:
        _ = _parse_box_any4(gt)
    except Exception:
        raise ValueError(f"Sample {idx}: ground_truth must be a dict with bbox+controls, got type {type(gt)}.")
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

def accuracy_reward_boxes(
    response: str,
    gt_bbox: Sequence[float],
) -> Tuple[float, float]:
    """
    Combined accuracy using:
        IoU_score    = exp((IoU * 2.5) - 1) / (e ** 2.5 - 1)   # Updated mapping with expK = 2.5
        acc          = IoU_score
    """
    # First, remove content before the first </think>
    response = _remove_content_before_think(response)

    # Parse the bounding box from the response using the custom format
    pred = _parse_boxed_from_start_end(response)
    if pred is None:
        return 0.0, 0.0

    # Calculate IoU between the predicted and ground truth bounding boxes
    iou = _iou_xyxy01(pred, gt_bbox)

    # New IoU -> score mapping with expK = 2.5
    denom = math.exp(2.5) - 1.0  # Using expK = 2.5 instead of 5
    iou_score = (math.exp((iou * 2.5) - 1.0) / denom) if denom > 0 else 0.0
    iou_score = float(max(0.0, min(1.0, iou_score)))

    return float(max(0.0, min(1.0, iou_score))), float(iou)

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
    Require shortcut expectation E in [0,1].
    Allowed sources (in order):
      1) validated.get("E")
      2) item["answer"]["E"]
      3) item["ground_truth"]["E"]
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
