#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, json, argparse, re
from typing import Any, Dict, List

TIMEOUT_PAT = re.compile(r"(time[\s_-]?out|timeout)", re.IGNORECASE)

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def load_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            yield line_no, json.loads(line)

def pick(obj: Dict[str, Any], key: str, i: int) -> Any:
    v = obj.get(key, None)
    if isinstance(v, list):
        return v[i] if i < len(v) else None
    return v

def contains_nonascii(x: Any) -> bool:
    if x is None:
        return False
    if isinstance(x, str):
        return any(ord(ch) > 127 for ch in x)
    if isinstance(x, list):
        return any(contains_nonascii(v) for v in x)
    if isinstance(x, dict):
        return any(contains_nonascii(v) for v in x.values())
    return False

def is_empty_like(pred: Any) -> bool:
    # 核心：pred 为空就算 empty
    if pred is None:
        return True
    if isinstance(pred, str) and pred.strip() == "":
        return True
    return False

def text_has_timeout(x: Any) -> bool:
    if x is None:
        return False
    if isinstance(x, str):
        return TIMEOUT_PAT.search(x) is not None
    # code 有时是 list[str] 或更复杂结构
    if isinstance(x, list):
        return any(text_has_timeout(v) for v in x)
    if isinstance(x, dict):
        return any(text_has_timeout(v) for v in x.values())
    return False

def write_jsonl(path: str, rows: List[Dict[str, Any]]):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out_dir", default="debug_evaluate")
    ap.add_argument("--max_keep_per_idx", type=int, default=0,
                    help=">0 时每个 idx 每类最多保留 N 条，避免输出太大。0=不限制")
    args = ap.parse_args()

    ensure_dir(args.out_dir)

    buckets = {
        "empty": [],
        "time_out": [],
        "empty_and_nonascii": [],
        "empty_only_ascii": [],
        "nonascii_only_not_empty": [],
    }
    per_idx = {k: {} for k in buckets}

    def cap_ok(bucket: str, idx_str: str) -> bool:
        if args.max_keep_per_idx <= 0:
            return True
        c = per_idx[bucket].get(idx_str, 0)
        if c >= args.max_keep_per_idx:
            return False
        per_idx[bucket][idx_str] = c + 1
        return True

    idx_summary: Dict[str, Dict[str, int]] = {}

    total_records = 0
    total_rollouts = 0

    for line_no, obj in load_jsonl(args.input):
        total_records += 1
        idx = obj.get("idx", None)
        idx_str = str(idx) if idx is not None else f"line_{line_no}"

        fr = obj.get("finish_reason", None)
        # 你这里 finish_reason 是 stop，但我们只用它来判断 list 长度
        n = len(fr) if isinstance(fr, list) else (len(obj["code"]) if isinstance(obj.get("code", None), list) else 1)

        idx_summary.setdefault(idx_str, {
            "line_no": line_no,
            "total_rollouts": 0,
            "empty_rollouts": 0,
            "timeout_like_rollouts": 0,
            "code_nonascii_rollouts": 0,
            "empty_and_nonascii_rollouts": 0,
            "empty_only_ascii_rollouts": 0,
            "nonascii_only_not_empty_rollouts": 0,
        })
        idx_summary[idx_str]["total_rollouts"] += n
        total_rollouts += n

        for rid in range(n):
            pred_i = pick(obj, "pred", rid)
            report_i = pick(obj, "report", rid)
            code_i = pick(obj, "code", rid)
            score_i = pick(obj, "score", rid)
            finish_reason_i = fr[rid] if isinstance(fr, list) else fr

            empty_like = is_empty_like(pred_i)
            timeout_like = text_has_timeout(report_i) or text_has_timeout(pred_i) or text_has_timeout(code_i)
            nonascii = contains_nonascii(code_i)

            if empty_like:
                idx_summary[idx_str]["empty_rollouts"] += 1
            if timeout_like:
                idx_summary[idx_str]["timeout_like_rollouts"] += 1
            if nonascii:
                idx_summary[idx_str]["code_nonascii_rollouts"] += 1

            row = {
                "line_no": line_no,
                "idx": idx,
                "rollout_id": rid,
                "finish_reason": finish_reason_i,
                "empty_like": empty_like,
                "timeout_like": timeout_like,
                "code_nonascii": nonascii,
                "question": obj.get("question", None),
                "gt": obj.get("gt", None),
                "gt_cot": obj.get("gt_cot", None),
                "answer": obj.get("answer", None),
                "pred": pred_i,
                "report": report_i,
                "score": score_i,
                "code": code_i,
            }

            # 输出分桶
            if empty_like and cap_ok("empty", idx_str):
                buckets["empty"].append(row)
            if timeout_like and cap_ok("time_out", idx_str):
                buckets["time_out"].append(row)

            if empty_like and nonascii:
                idx_summary[idx_str]["empty_and_nonascii_rollouts"] += 1
                if cap_ok("empty_and_nonascii", idx_str):
                    buckets["empty_and_nonascii"].append(row)
            elif empty_like and (not nonascii):
                idx_summary[idx_str]["empty_only_ascii_rollouts"] += 1
                if cap_ok("empty_only_ascii", idx_str):
                    buckets["empty_only_ascii"].append(row)
            elif (not empty_like) and nonascii:
                idx_summary[idx_str]["nonascii_only_not_empty_rollouts"] += 1
                if cap_ok("nonascii_only_not_empty", idx_str):
                    buckets["nonascii_only_not_empty"].append(row)

    # 写文件
    for name, rows in buckets.items():
        write_jsonl(os.path.join(args.out_dir, f"{name}.jsonl"), rows)

    with open(os.path.join(args.out_dir, "idx_summary.json"), "w", encoding="utf-8") as f:
        json.dump(idx_summary, f, ensure_ascii=False, indent=2)

    print(f"[DONE] records: {total_records}, rollouts scanned: {total_rollouts}")
    for name, rows in buckets.items():
        print(f"[DONE] {name}: {len(rows)} -> {os.path.join(args.out_dir, name + '.jsonl')}")
    print(f"[DONE] idx_summary -> {os.path.join(args.out_dir, 'idx_summary.json')}")

if __name__ == "__main__":
    main()
