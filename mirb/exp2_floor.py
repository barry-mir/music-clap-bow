"""Exp. 2, text prior: an LLM that never sees the audio picks between c+ and c-.

Qwen2.5-7B-Instruct is shown both captions (labelled A and B) with the fixed
prompt below and no audio; the choice is read from the next-token logits of
"A" and "B". Each pair is queried in both orders and the two probabilities are
averaged to cancel position bias. Accuracy above 0.5 measures how far language
priors alone go on the benchmark (the "LLM floor"). Writes results/floor.json
with per-pair choices, which mirb.summarize correlates with the audio models.

    python -m mirb.exp2_floor
    QWEN_DEVICE_MAP=auto python -m mirb.exp2_floor     # shard across GPUs
"""
import json
import os

import numpy as np
import torch
from scipy import stats
from transformers import AutoModelForCausalLM, AutoTokenizer

from mirb.data import load_pairs
from mirb.paths import RESULTS

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
PROMPT = (
    "You are given two short descriptions of a piece of music, labeled A and B. "
    "They use the same words but describe different musical arrangements; exactly "
    "one matches a real recording. You cannot hear the audio. Based only on which "
    "arrangement is more musically plausible, answer with the single letter A or B.\n"
    "A: {A}\nB: {B}\nAnswer:"
)


def main():
    pairs = load_pairs(with_audio=False)
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    dm = os.environ.get("QWEN_DEVICE_MAP", "cuda:0")
    kw = {"max_memory": {0: "9GiB", 1: "9GiB", "cpu": "30GiB"}} if dm == "auto" else {}
    lm = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float16, device_map=dm, **kw).eval()
    a_id = tok("A", add_special_tokens=False).input_ids[-1]
    b_id = tok("B", add_special_tokens=False).input_ids[-1]

    def p_first(cap_a, cap_b):
        msgs = [{"role": "user", "content": PROMPT.format(A=cap_a, B=cap_b)}]
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        ids = tok(text, return_tensors="pt").input_ids.to(lm.device)
        with torch.no_grad():
            logits = lm(ids).logits[0, -1, :].float()
        pa, _ = torch.softmax(torch.stack([logits[a_id], logits[b_id]]), 0).tolist()
        return pa

    rows = []
    for t in pairs:
        p_as_a = p_first(t["caption_pos"], t["caption_neg"])
        p_as_b = 1.0 - p_first(t["caption_neg"], t["caption_pos"])
        p_plus = 0.5 * (p_as_a + p_as_b)
        rows.append({"twin_id": t["pair_id"], "axis": t["axis"], "style": t["style"],
                     "p_plus": round(p_plus, 4), "picked_pos": int(p_plus > 0.5), "correct": int(p_plus > 0.5),
                     "position_bias": round(p_as_a - p_as_b, 4)})

    def summ(sub):
        k, n = sum(x["correct"] for x in sub), len(sub)
        return {"n": n, "acc": round(k / n, 4), "binom_p_vs_chance": round(stats.binomtest(k, n, 0.5).pvalue, 5)}

    report = {"model": MODEL_ID, "overall": summ(rows),
              "by_axis": {ax: summ([r for r in rows if r["axis"] == ax]) for ax in ("T", "R", "O")},
              "mean_position_bias": round(float(np.mean([r["position_bias"] for r in rows])), 4)}
    out = RESULTS / "floor.json"
    out.write_text(json.dumps({"report": report, "rows": rows}, indent=2))
    print(json.dumps(report, indent=2)); print("wrote", out)


if __name__ == "__main__":
    main()
