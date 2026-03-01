import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import math

MODEL_PATH = "/inspire/hdd/project/robot-dna/sujiadi-p-sujiadi/baibizhe-tmp/xinyue/verl/checkpoints/d0205_mQwen3-4B-Base_psnTrue_sigma0.0005_mods_.mlp.__seed20752_adaptFalse_tkl0.003_acoef1.01_ainit0.001_amin0.000001_amax0.5_tiscap10.0/global_step_250/actor/huggingface"  # 改成你的 merged 模型目录
DTYPE = torch.float16  # 或 bfloat16，按你实际

def tensor_stats(t):
    t = t.detach().float()
    return {
        "shape": list(t.shape),
        "min": float(t.min().item()),
        "max": float(t.max().item()),
        "mean": float(t.mean().item()),
        "std": float(t.std().item()),
        "nan": int(torch.isnan(t).any().item()),
        "inf": int(torch.isinf(t).any().item()),
    }

def main():
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=DTYPE,
        device_map="cpu",
        trust_remote_code=True
    )

    # 重点检查：embed、lm_head、norm、第一层/最后一层权重
    keys_to_check = []
    for name, p in model.named_parameters():
        if any(x in name.lower() for x in ["embed", "lm_head", "norm", "layernorm", "rmsnorm"]):
            keys_to_check.append(name)

    # 额外抽样一些层
    sample_names = []
    for name, _ in model.named_parameters():
        if ".0." in name or ".1." in name or "layers.0" in name or "layers.1" in name:
            sample_names.append(name)
    keys_to_check += sample_names[:20]

    seen = set()
    keys_to_check = [k for k in keys_to_check if not (k in seen or seen.add(k))]

    bad = 0
    for name in keys_to_check[:80]:
        p = dict(model.named_parameters())[name]
        st = tensor_stats(p)
        if st["nan"] or st["inf"] or abs(st["max"]) > 1000 or st["std"] > 100:
            bad += 1
            print("[BAD]", name, st)
    print(f"[DONE] checked={min(80,len(keys_to_check))}, bad={bad}")

if __name__ == "__main__":
    main()
