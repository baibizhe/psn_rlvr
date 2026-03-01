import json
from collections import defaultdict

INP = "debug_evaluate/questions_with_empty.jsonl"

def is_empty_pred(x):
    if x is None:
        return True
    if isinstance(x, str):
        return x.strip() == "" or x.strip().lower() in {"none", "null", "nan"}
    return False

def is_empty_code(x):
    if x is None:
        return True
    if isinstance(x, str):
        return x.strip() == ""
    if isinstance(x, (list, dict)):
        return len(x) == 0
    return False

per_idx = []

with open(INP, "r", encoding="utf-8") as f:
    for line_no, line in enumerate(f, 1):
        obj = json.loads(line)
        idx = obj.get("idx", line_no)

        preds = obj.get("pred", [])
        codes = obj.get("code", [])

        if not isinstance(preds, list):
            preds = [preds]
        if not isinstance(codes, list):
            codes = [codes] * len(preds)

        n = len(preds)
        a = b = pe = 0

        for i in range(n):
            pred_i = preds[i]
            code_i = codes[i] if i < len(codes) else None

            pred_empty = is_empty_pred(pred_i)
            code_empty = is_empty_code(code_i)

            if pred_empty:
                pe += 1
                if code_empty:
                    a += 1
                else:
                    b += 1

        per_idx.append({
            "idx": idx,
            "total": n,
            "pred_empty": pe,
            "pred_empty_code_empty": a,
            "pred_empty_code_nonempty": b,
            "pred_empty_ratio": pe / n if n else 0.0,
            "code_nonempty_when_pred_empty_ratio": (b / pe) if pe else 0.0,
        })

# 排序：最关心 pred_empty_code_nonempty（= 解析器/后处理清空的嫌疑最大）
per_idx_sorted = sorted(per_idx, key=lambda x: x["pred_empty_code_nonempty"], reverse=True)

print("[TOP 10] by pred_empty_code_nonempty:")
for r in per_idx_sorted[:10]:
    print(r)

# 全局统计
tot = sum(r["total"] for r in per_idx)
tot_pe = sum(r["pred_empty"] for r in per_idx)
tot_a = sum(r["pred_empty_code_empty"] for r in per_idx)
tot_b = sum(r["pred_empty_code_nonempty"] for r in per_idx)

print("\n[GLOBAL]")
print("total_rollouts:", tot)
print("pred_empty:", tot_pe)
print("pred_empty & code_empty:", tot_a)
print("pred_empty & code_nonempty:", tot_b)
print("ratio(code_nonempty | pred_empty):", (tot_b / tot_pe) if tot_pe else 0.0)
