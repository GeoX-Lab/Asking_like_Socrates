# Copyright 2024 Bytedance Ltd. and/or its affiliates
# ... 省略版权与license ...

import re
import math
from typing import Any, Optional

from mathruler.grader import extract_boxed_content, grade_answer


def format_reward(response: str) -> float:
    pattern = re.compile(r"<think>.*</think>.*\\boxed\{.*\}.*", re.DOTALL)
    format_match = re.fullmatch(pattern, response)
    return 1.0 if format_match else 0.0


import math
import re
from typing import Any, Optional
from mathruler.grader import extract_boxed_content

def _to_number(s: str) -> Optional[float]:
    if s is None:
        return None
    t = s.strip().replace(",", "")
    m = re.fullmatch(r"([-+]?\d+)\s*/\s*(\d+)", t)
    if m:
        num, den = int(m.group(1)), int(m.group(2))
        if den == 0:
            return None
        return num / den
    try:
        return float(t)
    except Exception:
        return None

def _ratio_expm1_stable(a: float, b: float, eps: float = 1e-9) -> float:
    """
    稳定计算 (exp(a)-1)/(exp(b)-1)，a,b >= 0
    - b 很小时，用泰勒近似：expm1(x) ~ x
    - 用 e^{a-b} * (1-e^{-a})/(1-e^{-b}) 避免溢出
    - 对 delta=a-b 做阈值裁剪，防止 e^{delta} 溢出
    """
    if b < 1e-6:
        # denom ≈ b，num ≈ a
        denom = max(b, eps)
        return a / denom

    delta = a - b
    if delta > 60:
        exp_delta = float('inf')   # 比例基本趋向于无穷大
    elif delta < -60:
        exp_delta = 0.0            # 比例基本为 0
    else:
        exp_delta = math.exp(delta)

    # e^{-x} 在 x 大时很安全，直接用
    exp_neg_a = 0.0 if a > 1e3 else math.exp(-a)
    exp_neg_b = 0.0 if b > 1e3 else math.exp(-b)

    denom2 = 1.0 - exp_neg_b
    if denom2 < eps:
        # 当 b 极小（但没走到 b<1e-6 分支）时的额外保护
        denom2 = max(b, eps)
        num2 = a
        return exp_delta * (num2 / denom2)

    num2 = 1.0 - exp_neg_a
    r = exp_delta * (num2 / denom2)
    return r

def accuracy_reward(response: str, ground_truth: str,
                    eps: float = 1e-9) -> float:
    """
    连续准确率：R = 1 - (exp(|x-k|)-1)/(exp(|k|)-1)
    数值稳定实现，避免 OverflowError。
    """
    answer_str = extract_boxed_content(response)
    x = _to_number(answer_str)
    k = _to_number(ground_truth)
    if x is None or k is None:
        return 0.0

    a = abs(x - k)         # |x - k|
    b = abs(k)             # 用 |k| 保证分母正且与原式一致
    r = _ratio_expm1_stable(a, b, eps=eps)   # (exp(a)-1)/(exp(b)-1)

    # R = 1 - r，并裁到 [0,1]
    score = 1.0 - r
    if score < 0.0:
        score = 0.0
    elif score > 1.0:
        score = 1.0
    return float(score)


def compute_score(reward_inputs: list[dict[str, Any]], format_weight: float = 0.1) -> list[dict[str, float]]:
    if not isinstance(reward_inputs, list):
        raise ValueError("Please use `reward_type=batch` for math reward function.")

    scores = []
    for reward_input in reward_inputs:
        response = re.sub(r"\s*(<|>|/)\s*", r"\1", reward_input["response"])  # handle qwen2.5vl-32b format
        format_score = format_reward(response)
        accuracy_score = accuracy_reward(response, reward_input["ground_truth"])
        scores.append(
            {
                "overall": (1 - format_weight) * accuracy_score + format_weight * format_score,
                "format": format_score,
                "accuracy": accuracy_score,
            }
        )

    return scores
