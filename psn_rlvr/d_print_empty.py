import json, random

INP = "debug_evaluate/questions_with_empty.jsonl"
TARGET_IDX = 11
K = 10

def is_empty_pred(x):
    return x is None or (isinstance(x, str) and x.strip() == "")

def is_nonempty_str(x):
    return isinstance(x, str) and x.strip() != ""

rows = []
with open(INP, "r", encoding="utf-8") as f:
    for line in f:
        obj = json.loads(line)
        if obj.get("idx") == TARGET_IDX:
            preds = obj.get("pred", [])
            codes = obj.get("code", [])
            for i in range(len(preds)):
                pred_i = preds[i]
                code_i = codes[i] if i < len(codes) else None
                if is_empty_pred(pred_i) and is_nonempty_str(code_i):
                    rows.append((i, code_i))
            break

print(f"[STAT] idx={TARGET_IDX} empty_pred & code_nonempty count = {len(rows)}")
random.seed(0)
for rid, code in random.sample(rows, k=min(K, len(rows))):
    print("\n" + "="*80)
    print(f"rollout_id={rid}")
    print(code[:1200])
