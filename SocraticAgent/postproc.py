import argparse
import json
import logging
import re
import uuid
from typing import Any, Dict, List, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

# ------- Regex Patterns -------
THINKING_RE = re.compile(r"<thinking>\s*(.*?)\s*</thinking>", re.DOTALL | re.IGNORECASE)
QUESTION_RE = re.compile(r"<question>\s*(.*?)\s*</question>", re.DOTALL | re.IGNORECASE)


def _normalize_ws(s: str) -> str:
    """Collapses multiple whitespace characters into a single space."""
    return re.sub(r"\s+", " ", s).strip()


def _extract_first(pattern: re.Pattern, text: Optional[str]) -> str:
    """Extracts the first regex group match from text."""
    if not text:
        return ""
    m = pattern.search(text)
    return m.group(1).strip() if m else ""


def _detect_lang(item: Dict[str, Any]) -> str:
    """Detects language ('zh' or 'en') from item metadata."""
    lang = str(item.get("lang", "")).strip().lower()
    return "zh" if lang.startswith("zh") or lang == "cn" else "en"


def build_self_talk_merged(item: Dict[str, Any]) -> str:
    """
    Constructs the chain-of-thought string matching the dataset format:
    <think>{header}{round1_think}{q1}{p1}... </think>{final_answer}
    """
    loop = item.get("loop_result", {})
    final_answer = loop.get("final_answer", "")
    chat_history: List[Dict[str, Any]] = loop.get("chat_history", [])

    lang = _detect_lang(item)
    
    # Define header based on language
    header = (
        "好的，我将会以自问自答的方式逐步推理，最终给出回复。 "
        if lang == "zh"
        else "Alright, I will reason in a self Q&A style and give the final reply. "
    )

    segments: List[str] = []

    for i, turn in enumerate(chat_history, start=1):
        r_resp = str(turn.get("R_response", "")).strip()

        # 1. Extract thinking block
        thinking = _extract_first(THINKING_RE, r_resp)
        if thinking:
            segments.append(thinking)

        # 2. Extract question (prefer specific input key, fallback to regex)
        q_text = str(turn.get("P_query_input", "")).strip()
        if not q_text:
            q_text = _extract_first(QUESTION_RE, r_resp)
        if q_text:
            segments.append(_normalize_ws(q_text))

        # 3. Append perception response
        p_resp = str(turn.get("P_response", "")).strip()
        if p_resp:
            segments.append(p_resp)

    # Combine: <think>header + segments</think> + final_answer
    think_body = " ".join(segments)
    merged = f"<think>{header}{think_body}</think>"
    
    if final_answer:
        merged += f"{final_answer}"
    
    return merged


def load_jsonl(file_path: str) -> List[Dict[str, Any]]:
    """Loads a JSONL file into a list of dictionaries."""
    data = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_no, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError as e:
                    logger.warning(f"Skipping invalid JSON at line {line_no}: {e}")
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        exit(1)
    return data


def process_dataset(input_path: str, output_path: str):
    """Processes the dataset, filtering for success and formatting output."""
    logger.info(f"Loading data from {input_path}...")
    data = load_jsonl(input_path)

    results = []
    skipped_count = 0

    for item in data:
        # Filter: Skip if explicitly marked as error
        if item.get("error"):
            skipped_count += 1
            continue

        loop = item.get("loop_result", {})
        
        # Filter: Require strict success and a final answer
        if not loop.get("success") or not loop.get("final_answer"):
            skipped_count += 1
            continue

        merged_thought = build_self_talk_merged(item)
        
        results.append({
            "id": str(uuid.uuid4()),
            "query": item.get("rewritten_query"),
            "raw_query": item.get("query"),
            "answer": loop.get("final_answer"),
            "raw_gt": item.get("gt"),
            "thinking": merged_thought,
            "image": item.get("image"),
            "image_root": item.get("image_root"),
            "data_source": item.get("data_source"),
            "task": item.get("task")
        })

    logger.info(f"Processed {len(results)} valid items. Skipped {skipped_count} items.")
    
    logger.info(f"Saving to {output_path}...")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Merge Socratic Loop JSONL results into training format.")
    parser.add_argument("--data_path", type=str, required=True, help="Path to input .jsonl file.")
    parser.add_argument("--out_path", type=str, default=None, help="Path to output .json file.")
    
    args = parser.parse_args()

    # Determine output path if not provided
    if not args.out_path:
        args.out_path = args.data_path.replace(".jsonl", "_merge.json")

    process_dataset(args.data_path, args.out_path)


if __name__ == "__main__":
    main()