"""Exp. 2, text prior: a model that never sees the audio picks between c+ and c-.

The model is shown both captions (labelled A and B) with the fixed prompt below
and no audio; the choice is read from the next-token logits of "A" and "B". Each
pair is queried in both orders and the two probabilities are averaged to cancel
position bias. Accuracy above 0.5 measures how far language priors alone go.

Two baselines:
  --model lalm  Qwen2-Audio-7B-Instruct run without audio. This is the paper's
                Exp. 2: the LALM's own text prior, directly comparable with its
                with-audio choices from Exp. 1.  -> results/floor_lalm.json
  --model llm   Qwen2.5-7B-Instruct, an external text-only reference.
                -> results/floor.json

    python -m masb.exp2_floor                        # the paper's Exp. 2 (lalm)
    QWEN_DEVICE_MAP=auto python -m masb.exp2_floor   # shard across GPUs
"""
import argparse
import json
import os

import numpy as np
import torch
from scipy import stats

from masb.data import load_pairs
from masb.paths import RESULTS

MODEL_ID = {"lalm": "Qwen/Qwen2-Audio-7B-Instruct", "llm": "Qwen/Qwen2.5-7B-Instruct"}
PROMPT = (
    "You are given two short descriptions of a piece of music, labeled A and B. "
    "They use the same words but describe different musical arrangements; exactly "
    "one matches a real recording. You cannot hear the audio. Based only on which "
    "arrangement is more musically plausible, answer with the single letter A or B.\n"
    "A: {A}\nB: {B}\nAnswer:"
)


def load_model(which):
    """Returns (tokenizer, model, chat_template_fn). The LALM is loaded without any
    audio input, so only its language model is exercised."""
    mid = MODEL_ID[which]
    dm = os.environ.get("QWEN_DEVICE_MAP", "cuda:0")
    kw = {"max_memory": {0: "9GiB", 1: "9GiB", "cpu": "30GiB"}} if dm == "auto" else {}
    if which == "lalm":
        from transformers import Qwen2AudioForConditionalGeneration, AutoProcessor
        proc = AutoProcessor.from_pretrained(mid)
        lm = Qwen2AudioForConditionalGeneration.from_pretrained(
            mid, torch_dtype=torch.float16, device_map=dm, **kw).eval()
        tok = proc.tokenizer

        def chat(text):  # the LALM needs the content-list form of the chat template
            conv = [{"role": "user", "content": [{"type": "text", "text": text}]}]
            return proc.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)
    else:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tok = AutoTokenizer.from_pretrained(mid)
        lm = AutoModelForCausalLM.from_pretrained(
            mid, torch_dtype=torch.float16, device_map=dm, **kw).eval()

        def chat(text):
            return tok.apply_chat_template([{"role": "user", "content": text}],
                                           tokenize=False, add_generation_prompt=True)
    return tok, lm, chat


def main(which="lalm"):
    pairs = load_pairs(with_audio=False)
    tok, lm, chat = load_model(which)
    dev = next(lm.parameters()).device
    a_id = tok("A", add_special_tokens=False).input_ids[-1]
    b_id = tok("B", add_special_tokens=False).input_ids[-1]

    def p_first(cap_a, cap_b):
        ids = tok(chat(PROMPT.format(A=cap_a, B=cap_b)), return_tensors="pt").input_ids.to(dev)
        with torch.no_grad():
            logits = lm(input_ids=ids).logits[0, -1, :].float()
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

    report = {"model": MODEL_ID[which] + (" (text only)" if which == "lalm" else ""), "overall": summ(rows),
              "by_axis": {ax: summ([r for r in rows if r["axis"] == ax]) for ax in ("T", "R", "O")},
              "mean_position_bias": round(float(np.mean([r["position_bias"] for r in rows])), 4)}
    out = RESULTS / ("floor_lalm.json" if which == "lalm" else "floor.json")
    out.write_text(json.dumps({"report": report, "rows": rows}, indent=2))
    print(json.dumps(report, indent=2)); print("wrote", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="lalm", choices=["lalm", "llm"],
                    help="lalm: Qwen2-Audio without audio (the paper's Exp. 2); llm: Qwen2.5 reference")
    a = ap.parse_args()
    main(a.model)
