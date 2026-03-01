import os, json, re
from collections import defaultdict

INPUT_JSONL = "/inspire/hdd/project/robot-dna/sujiadi-p-sujiadi/baibizhe-tmp/xinyue/verl/checkpoints/d0205_mQwen3-4B-Base_psnTrue_sigma0.0005_mods_.mlp.__seed20752_adaptFalse_tkl0.003_acoef1.01_ainit0.001_amin0.000001_amax0.5_tiscap10.0/eval_results_passk_temp0.9_qwen3-thinking/global_step_250/aime25/test_qwen3-thinking_-1_seed0_t0.9_s0_e-1.jsonl"
OUT_DIR = "debug_evaluate"

TIMEOUT_PAT = re.compile(
    r"(time\s*out|timeout|timed\s*out|tle|time\s*limit|deadline|watchdog|exceeded\s*time|took\s*too\s*long)",
    re.I
)

def is_empty_text(x):
    return x is None or (isinstance(x, str) and x.strip() == "")

def non_ascii_ratio(s: str) -> float:
    if not s:
        return 0.0
    non_ascii = sum(1 for ch in s if ord(ch) > 127)
    return non_ascii / max(1, len(s))

def has_timeout_keyword(code):
    return isinstance(code, str) and bool(TIMEOUT_PAT.search(code))

os.makedirs(OUT_DIR, exist_ok=True)

summary = {}  # idx -> stats
timeout_idx = set()
empty_idx = set()
nonascii_idx = set()

# 如果你想把“出现过异常的题目”整题导出，打开下面开关
EXPORT_FULL_QUESTIONS = True
full_timeout_questions = []  # 每条是一整题（保留原结构）
full_empty_questions = []

with open(INPUT_JSONL, "r", encoding="utf-8") as f:
    for line_no, line in enumerate(f, 1):
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)

        idx = obj.get("idx")
        preds = obj.get("pred", [])
        codes = obj.get("code", [])

        if not isinstance(preds, list):
            preds = [preds]
        if not isinstance(codes, list):
            codes = [codes] * len(preds)

        total = len(preds)
        empty_cnt = 0
        timeout_cnt = 0
        nonascii_cnt = 0

        for i in range(total):
            pred_i = preds[i]
            code_i = codes[i] if i < len(codes) else None

            if is_empty_text(pred_i):
                empty_cnt += 1
            if has_timeout_keyword(code_i):
                timeout_cnt += 1
            if isinstance(code_i, str) and non_ascii_ratio(code_i) > 0.20:
                nonascii_cnt += 1

        summary[idx] = {
            "line_no": line_no,
            "total_rollouts": total,
            "empty_rollouts": empty_cnt,
            "timeout_rollouts": timeout_cnt,
            "code_nonascii_rollouts": nonascii_cnt,
        }

        if empty_cnt > 0:
            empty_idx.add(idx)
            if EXPORT_FULL_QUESTIONS:
                full_empty_questions.append(obj)
        if timeout_cnt > 0:
            timeout_idx.add(idx)
            if EXPORT_FULL_QUESTIONS:
                full_timeout_questions.append(obj)
        if nonascii_cnt > 0:
            nonascii_idx.add(idx)

# 写 summary
with open(os.path.join(OUT_DIR, "idx_summary.json"), "w", encoding="utf-8") as wf:
    json.dump(summary, wf, ensure_ascii=False, indent=2)

# 写 idx 列表
def dump_idx_list(name, idx_set):
    with open(os.path.join(OUT_DIR, name), "w", encoding="utf-8") as wf:
        for x in sorted(idx_set, key=lambda z: (str(type(z)), z)):
            wf.write(str(x) + "\n")

dump_idx_list("idx_timeout_list.txt", timeout_idx)
dump_idx_list("idx_empty_list.txt", empty_idx)
dump_idx_list("idx_code_nonascii_list.txt", nonascii_idx)

# 可选：整题导出（30条以内，非常方便逐题复盘）
if EXPORT_FULL_QUESTIONS:
    with open(os.path.join(OUT_DIR, "questions_with_timeout.jsonl"), "w", encoding="utf-8") as wf:
        for obj in full_timeout_questions:
            wf.write(json.dumps(obj, ensure_ascii=False) + "\n")
    with open(os.path.join(OUT_DIR, "questions_with_empty.jsonl"), "w", encoding="utf-8") as wf:
        for obj in full_empty_questions:
            wf.write(json.dumps(obj, ensure_ascii=False) + "\n")

print(f"[DONE] idx_summary -> {OUT_DIR}/idx_summary.json")
print(f"[DONE] timeout idx count: {len(timeout_idx)}  -> {OUT_DIR}/idx_timeout_list.txt")
print(f"[DONE] empty idx count:   {len(empty_idx)}  -> {OUT_DIR}/idx_empty_list.txt")
print(f"[DONE] nonascii idx count:{len(nonascii_idx)} -> {OUT_DIR}/idx_code_nonascii_list.txt")
if EXPORT_FULL_QUESTIONS:
    print(f"[DONE] questions_with_timeout -> {OUT_DIR}/questions_with_timeout.jsonl")
    print(f"[DONE] questions_with_empty   -> {OUT_DIR}/questions_with_empty.jsonl")
