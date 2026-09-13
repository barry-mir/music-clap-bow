"""Every number reported in the paper, from the results JSONs -> results/summary.json.

    python -m masb.stats        # first: Wilson CIs, binomial p, Holm over the 15 model x property tests
    python -m masb.summarize    # then this

Adds to stats.json:
  kappa          Cohen's kappa between each model's per-pair choice and the LLM's (Exp. 2)
  qwen_prior     Qwen2-Audio accuracy on prior-consistent / prior-inconsistent pairs
                 (split by whether the LLM was right) and the prior-balanced mean
  text_medians   Exp. 3 median distances per encoder and the swap > paraphrase rate
  order          Exp. 4 symmetric accuracy per model (from results/order/*.json)
"""
import glob
import json

import numpy as np
from scipy import stats as st

from masb.paths import RESULTS

MODELS = ["laion-clap-music", "ms-clap-2023", "muq-mulan", "clamp3-saas", "qwen2-audio-7b-instruct"]
TEXT = {"laion-clap": "laion-clap-music", "ms-clap": "ms-clap-2023", "muq-mulan": "muq-mulan", "clamp3": "clamp3-saas"}


def kappa(m, l):
    m, l = np.asarray(m), np.asarray(l)
    po = float((m == l).mean()); pm, pl = m.mean(), l.mean()
    pe = pm * pl + (1 - pm) * (1 - pl)
    return float((po - pe) / (1 - pe)) if pe < 1 else float("nan")


def main():
    stats = json.load(open(RESULTS / "stats.json"))
    floor = {(r["twin_id"], r["style"]): r for r in json.load(open(RESULTS / "floor.json"))["rows"]}
    evals = {}
    for f in glob.glob(str(RESULTS / "eval_*.json")):
        d = json.load(open(f)); evals[d["report"]["model"]] = d["rows"]
    out = {"swap_accuracy": stats["audio"], "holm": stats["holm_swap_acc_vs_chance"], "floor": stats["floor"],
           "kappa": {}, "qwen_prior": {}, "text_medians": {}, "order": {}}

    for m in MODELS:
        if m not in evals:
            continue
        out["kappa"][m] = {}
        for ax in "TRO":
            rs = [r for r in evals[m] if r["axis"] == ax]
            out["kappa"][m][ax] = round(kappa([int(r["s_pos"] > r["s_neg"]) for r in rs],
                                              [floor[(r["twin_id"], r["style"])]["picked_pos"] for r in rs]), 3)

    q = evals.get("qwen2-audio-7b-instruct")
    if q:
        for ax in "TRO":
            rs = [r for r in q if r["axis"] == ax]
            correct = [int(r["s_pos"] > r["s_neg"]) for r in rs]
            llm_ok = [floor[(r["twin_id"], r["style"])]["correct"] for r in rs]
            cons = [c for c, f in zip(correct, llm_ok) if f == 1]
            inc = [c for c, f in zip(correct, llm_ok) if f == 0]
            ac, ai = float(np.mean(cons)), float(np.mean(inc))
            out["qwen_prior"][ax] = {"prior_consistent_acc": round(ac, 3), "n_consistent": len(cons),
                                     "prior_inconsistent_acc": round(ai, 3), "n_inconsistent": len(inc),
                                     "inconsistent_binom_p": round(float(st.binomtest(int(sum(inc)), len(inc), 0.5).pvalue), 4),
                                     "prior_balanced_acc": round((ac + ai) / 2, 3)}

    for key, name in TEXT.items():
        f = RESULTS / f"text_text_{key}.json"
        if not f.exists():
            continue
        rs = json.load(open(f))["rows"]
        out["text_medians"][name] = {k: round(float(np.median([r[k] for r in rs])), 4) for k in ("d_swap", "d_paraphrase", "d_other")}
        out["text_medians"][name]["swap_gt_paraphrase_rate"] = round(float(np.mean([r["text_binding_correct"] for r in rs])), 4)

    for f in glob.glob(str(RESULTS / "order" / "order_*.json")):
        d = json.load(open(f))
        out["order"][d["model"]] = {"symmetric": d["symmetric"], "mirror_view": d["mirror_view"],
                                    **({"audio_distance": d["audio_distance"]} if "audio_distance" in d else {})}

    (RESULTS / "summary.json").write_text(json.dumps(out, indent=1))
    # short console table
    print("swap accuracy (T / R / O / all):")
    for m in MODELS:
        if m in out["swap_accuracy"]:
            a = out["swap_accuracy"][m]
            print(f"  {m:26s} " + "  ".join(f"{a[ax]['acc']:.3f}" for ax in ("T", "R", "O", "ALL")))
    print("LLM floor: " + "  ".join(f"{out['floor'][ax]['acc']:.3f}" for ax in ("T", "R", "O", "ALL")))
    print("kappa vs LLM:", json.dumps(out["kappa"]))
    print("Qwen prior-balanced:", {ax: v["prior_balanced_acc"] for ax, v in out["qwen_prior"].items()})
    print("Exp. 4 symmetric:", {m: v["symmetric"]["acc"] for m, v in out["order"].items()})
    print("wrote", RESULTS / "summary.json")


if __name__ == "__main__":
    main()
