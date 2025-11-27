#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
无裁剪版本验证：
- 被测 compute_score 不裁剪 r^E 与 overall（按你提供的当前实现）
- 修正了 T2/T4/T5 的期望计算

运行:
  python verify_compute_score_noclip.py
"""

import os
import importlib.util
from typing import Dict, Any, List, Tuple

# --------- 动态加载被测模块 ---------
MODULE_PATH = "/g0001sr/lzy/EasyR1/examples/reward_function/oursIOU_k0.py"
spec = importlib.util.spec_from_file_location("oursIOU_k0", MODULE_PATH)
if spec is None or spec.loader is None:
    raise RuntimeError(f"无法加载模块：{MODULE_PATH}")
ours = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ours)  # type: ignore

compute_score = ours.compute_score  # 被测函数

# --------- 独立 IoU 实现（用于核对）---------
def _fix_clip_xyxy01(box: Tuple[float, float, float, float]) -> Tuple[float, float, float, float]:
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

def iou_xyxy01(a: Tuple[float, float, float, float],
               b: Tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = _fix_clip_xyxy01(a)
    bx1, by1, bx2, by2 = _fix_clip_xyxy01(b)
    iw = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    ih = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return 0.0 if union <= 0 else inter / union

# --------- 构造样本工具 ---------
def mk_item(
    gt_bbox: Tuple[float, float, float, float],
    response: str,
    E: float,
    E_in: str = "answer",  # "answer" 或 "ground_truth"
) -> Dict[str, Any]:
    item: Dict[str, Any] = {
        "ground_truth": {"bbox": list(gt_bbox)},
        "response": response,
    }
    if E_in == "answer":
        item["answer"] = {"E": float(E)}
    else:
        item["ground_truth"]["E"] = float(E)
    return item

# --------- 断言工具 ---------
def almost_equal(a: float, b: float, eps: float = 1e-9) -> bool:
    return abs(a - b) <= eps

def check(cond: bool, msg: str, failures: List[str]):
    if not cond:
        failures.append(msg)

# --------- 响应模板 ---------
def resp_perfect_after_think_and_answer(bbox: Tuple[float, float, float, float]) -> str:
    x1, y1, x2, y2 = bbox
    return (
        "<think>some chain of thought ...</think>\n"
        "Answer: {\"bbox\": [" f"{x1}, {y1}, {x2}, {y2}" "]}"
    )

def resp_after_think_no_answer_with_bbox(bbox: Tuple[float, float, float, float]) -> str:
    x1, y1, x2, y2 = bbox
    return (
        "<think>...</think>\n"
        "{ \"bbox\": [" f"{x1}, {y1}, {x2}, {y2}" "] }"
    )

def resp_after_think_answer_fallback_any4(bbox: Tuple[float, float, float, float]) -> str:
    # 无 "bbox:" 键，仅方括号 -> 触发 fallback
    x1, y1, x2, y2 = bbox
    return (
        "<think>...</think>\n"
        "Answer: \n"
        "[" f"{x1}, {y1}, {x2}, {y2}" "]"
    )

def resp_early_bbox_before_think_then_normal_after(bbox: Tuple[float, float, float, float]) -> str:
    # 注意：为匹配你当前 _has_any_bbox_before 的 regex，这里用**不带引号**的 bbox 键
    x1, y1, x2, y2 = bbox
    return (
        "Prelude text ... Answer: {bbox: [0,0,1,1]}\n"  # 早产（无引号的 bbox）
        "<think>...</think>\n"
        "Answer: {\"bbox\": [" f"{x1}, {y1}, {x2}, {y2}" "]}"
    )

def resp_no_think_with_answer(bbox: Tuple[float, float, float, float]) -> str:
    x1, y1, x2, y2 = bbox
    return "Answer: {\"bbox\": [" f"{x1}, {y1}, {x2}, {y2}" "]}"

# --------- 主测试 ---------
def run_tests() -> None:
    failures: List[str] = []
    tests_run = 0

    gt = (100.0, 100.0, 400.0, 400.0)
    fw_default = 0.1  # format_weight

    # T1) IoU=1, E=0.3, 完整格式 -> overall = (1-fw)*1 + fw*1 = 1
    resp = resp_perfect_after_think_and_answer(gt)
    out = compute_score([mk_item(gt, resp, E=0.3)], format_weight=fw_default)[0]
    tests_run += 1
    check(almost_equal(out["accuracy_raw"], 1.0), "T1 accuracy_raw!=1", failures)
    check(almost_equal(out["format_total"], 1.0), "T1 format_total!=1", failures)
    check(almost_equal(out["accuracy"], 1.0), "T1 accuracy!=1", failures)
    check(almost_equal(out["overall"], 1.0), "T1 overall!=1", failures)

    # T2) IoU=0, E=0.3, 完整格式 -> acc_adj = (0-E)/(1-E) = -0.428571..., overall = (1-fw)*acc_adj + fw*1
    resp = resp_perfect_after_think_and_answer((0.0, 0.0, 1.0, 1.0))
    out = compute_score([mk_item(gt, resp, E=0.3)], format_weight=fw_default)[0]
    tests_run += 1
    expected_acc = (0.0 - 0.3) / (1.0 - 0.3)
    expected_overall = (1 - fw_default) * expected_acc + fw_default * 1.0
    check(almost_equal(out["accuracy_raw"], 0.0), "T2 accuracy_raw!=0", failures)
    check(almost_equal(out["accuracy"], expected_acc), "T2 accuracy^E mismatch (no clipping)", failures)
    check(almost_equal(out["format_total"], 1.0), "T2 format_total!=1", failures)
    check(almost_equal(out["overall"], expected_overall), "T2 overall mismatch (no clipping)", failures)
    #print("T2 debug:", out, "fw=", fw_default, "expected_overall=", expected_overall)

    # T3) IoU=partial, E=0.2, 完整格式 -> 按无裁剪公式计算
    half_overlap = (100.0, 100.0, 400.0, 250.0)
    iou = iou_xyxy01(half_overlap, gt)
    resp = resp_perfect_after_think_and_answer(half_overlap)
    out = compute_score([mk_item(gt, resp, E=0.2)], format_weight=fw_default)[0]
    tests_run += 1
    expected_acc = (iou - 0.2) / (1.0 - 0.2)
    expected_overall = (1 - fw_default) * expected_acc + fw_default * 1.0
    check(almost_equal(out["accuracy_raw"], iou), "T3 accuracy_raw!=IoU", failures)
    check(almost_equal(out["accuracy"], expected_acc), "T3 accuracy^E mismatch", failures)
    check(almost_equal(out["overall"], expected_overall), "T3 overall mismatch", failures)

    # T4) 回退解析：Answer 后只有 [a,b,c,d]，基础格式分 2/3，再 *0.5 => 1/3
    resp = resp_after_think_answer_fallback_any4(gt)
    out = compute_score([mk_item(gt, resp, E=0.3)], format_weight=fw_default)[0]
    tests_run += 1
    check(almost_equal(out["accuracy_raw"], 1.0), "T4 accuracy_raw!=1", failures)
    # format: thinkclose=1, answer=1, bbox_after=0 -> (1+1+0)/3 = 2/3; fallback 再 *0.5 => 1/3
    check(almost_equal(out["format_total"], (2.0/3.0) * 0.5), "T4 format_total != 1/3 under fallback", failures)
    check(almost_equal(out["used_fallback_any4"], 1.0), "T4 used_fallback_any4!=1", failures)

    # T5) 早产 bbox：think 前出现 {bbox:[...]}（无引号），-> format_total 对半
    resp = resp_early_bbox_before_think_then_normal_after(gt)
    out = compute_score([mk_item(gt, resp, E=0.0)], format_weight=fw_default)[0]
    tests_run += 1
    # 正常应是完整格式 1，然后因早产 *0.5；未触发 fallback，不再额外对半
    check(almost_equal(out["format_total"], 0.5), "T5 format_total should be halved by early-bbox penalty", failures)

    # T6) 无 </think>：accuracy_raw=0，format_total=0，总分=0
    resp = resp_no_think_with_answer(gt)
    out = compute_score([mk_item(gt, resp, E=0.0)], format_weight=fw_default)[0]
    tests_run += 1
    check(almost_equal(out["accuracy_raw"], 0.0), "T6 accuracy_raw should be 0 when no </think>", failures)
    check(almost_equal(out["format_total"], 0.0), "T6 format_total should be 0 when no </think>", failures)
    check(almost_equal(out["overall"], 0.0), "T6 overall should be 0 when no </think>", failures)

    # T7) 有 </think> 但无 Answer:；tail 里直接有 {"bbox":[]} -> accuracy>0 但 format 仅 1/3
    resp = resp_after_think_no_answer_with_bbox(gt)
    out = compute_score([mk_item(gt, resp, E=0.0)], format_weight=fw_default)[0]
    tests_run += 1
    check(almost_equal(out["accuracy_raw"], 1.0), "T7 accuracy_raw!=1", failures)
    check(almost_equal(out["format_total"], (1.0 + 0.0 + 0.0) / 3.0), "T7 format_total should be 1/3", failures)

    # T8) E->1：accuracy 仅当 accuracy_raw==1 才为 1，否则 0（你的实现就这么写的）
    #   8a) IoU=1, E=1
    resp = resp_perfect_after_think_and_answer(gt)
    out = compute_score([mk_item(gt, resp, E=1.0)], format_weight=0.0)[0]
    tests_run += 1
    check(almost_equal(out["accuracy_raw"], 1.0), "T8a accuracy_raw!=1", failures)
    check(almost_equal(out["accuracy"], 1.0), "T8a accuracy should be 1 when E=1 and acc_raw=1", failures)
    #   8b) IoU<1, E=1 -> accuracy=0
    near = (100.0, 100.0, 399.0, 399.0)
    resp = resp_perfect_after_think_and_answer(near)
    out = compute_score([mk_item(gt, resp, E=1.0)], format_weight=0.0)[0]
    tests_run += 1
    check(out["accuracy_raw"] < 1.0, "T8b accuracy_raw should be <1", failures)
    check(almost_equal(out["accuracy"], 0.0), "T8b accuracy should be 0 when E=1 and acc_raw<1", failures)

    # T9) format_weight 边界：fw=0 -> overall==acc_adj（无裁剪）；fw=1 -> overall==format_total
    half_overlap = (100.0, 100.0, 400.0, 250.0)
    iou_mid = iou_xyxy01(half_overlap, gt)
    resp = resp_perfect_after_think_and_answer(half_overlap)
    E = 0.2
    item = mk_item(gt, resp, E=E)
    out0 = compute_score([item], format_weight=0.0)[0]
    out1 = compute_score([item], format_weight=1.0)[0]
    tests_run += 1
    expected_acc = (iou_mid - E) / (1.0 - E)
    check(almost_equal(out0["overall"], expected_acc), "T9 fw=0 overall!=accuracy (no clipping)", failures)
    check(almost_equal(out1["overall"], 1.0), "T9 fw=1 overall!=format_total(=1)", failures)

    # 汇总
    if failures:
        print("❌ 部分测试未通过：")
        for i, msg in enumerate(failures, 1):
            print(f"  [{i}] {msg}")
    else:
        print(f"✅ 全部 {tests_run} 项测试通过。")

if __name__ == "__main__":
    if not os.path.isfile(MODULE_PATH):
        raise FileNotFoundError(f"找不到被测文件：{MODULE_PATH}")
    run_tests()
