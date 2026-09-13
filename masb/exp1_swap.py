"""Exp. 1, contrastive models: CLAP score under the attribute swap.

Per caption pair and model:
    s_pos  = cos(audio, c+)          s_neg   = cos(audio, c-)
    s_null = cos(audio, paraphrase)  s_wrong = cos(audio, another recording's c+)
A pair is correct when s_pos > s_neg; chance is 0.5. Writes results/eval_<model>.json
with one row per pair (raw scores) plus a summary; run masb.stats for CIs, Holm, etc.

    python -m masb.exp1_swap --model laion-clap      # ms-clap | muq-mulan | clamp3
"""
import argparse
import json

import numpy as np
from scipy import stats

from masb.data import load_pairs
from masb.models import REGISTRY
from masb.paths import RESULTS


def binom_p(k, n):
    return stats.binomtest(k, n, 0.5).pvalue if n else 1.0


def summarize(tag, gaps):
    g = np.asarray(gaps)
    if len(g) == 0:
        return {"tag": tag, "n": 0}
    k = int((g > 0).sum())
    return {"tag": tag, "n": len(g), "acc": round(k / len(g), 4), "binom_p": round(binom_p(k, len(g)), 5),
            "swap_gap_mean": round(float(g.mean()), 5), "swap_gap_std": round(float(g.std()), 5)}


def run(model_key, axis=None):
    pairs = load_pairs(axis)
    model = REGISTRY[model_key]()

    # one embedding per clip file; keyed by path because each (recording, property) has its own clip
    uris = sorted({r["audio_uri"] for r in pairs})
    aemb = dict(zip(uris, model.embed_audio(uris)))
    E_pos = model.embed_text([r["caption_pos"] for r in pairs])
    E_neg = model.embed_text([r["caption_neg"] for r in pairs])
    E_par = model.embed_text([r["caption_paraphrase"] for r in pairs])

    rows = []
    n = len(pairs)
    for i, r in enumerate(pairs):
        a = aemb[r["audio_uri"]]
        j = (i + n // 2) % n                       # a different recording's caption as the wrong-content anchor
        while pairs[j]["clip_id"] == r["clip_id"] and n > 1:
            j = (j + 1) % n
        s_pos, s_neg = float(a @ E_pos[i]), float(a @ E_neg[i])
        s_nul, s_wrong = float(a @ E_par[i]), float(a @ E_pos[j])
        rows.append({"twin_id": r["pair_id"], "clip_id": r["clip_id"], "axis": r["axis"], "style": r["style"],
                     "prior_class": r["prior_class"],
                     "s_pos": s_pos, "s_neg": s_neg, "s_null": s_nul, "s_wrong": s_wrong,
                     "swap_gap": s_pos - s_neg, "null_gap": s_pos - s_nul, "wrong_gap": s_pos - s_wrong})

    swap = [x["swap_gap"] for x in rows]
    try:
        w_stat, w_p = stats.wilcoxon(swap, [x["null_gap"] for x in rows])
    except ValueError:
        w_stat, w_p = float("nan"), float("nan")
    breakdown = {}
    for key in ("axis", "style", "prior_class"):
        for val in sorted({x[key] for x in rows}):
            breakdown[f"{key}={val}"] = summarize(f"{key}={val}", [x["swap_gap"] for x in rows if x[key] == val])
    report = {"model": model.name, "n_pairs": n, "overall": summarize("overall", swap),
              "anchors": {"null_gap_mean": round(float(np.mean([x["null_gap"] for x in rows])), 5),
                          "wrong_gap_mean": round(float(np.mean([x["wrong_gap"] for x in rows])), 5)},
              "wilcoxon_swap_vs_null": {"stat": float(w_stat), "p": float(w_p)}, "breakdown": breakdown}
    out = RESULTS / f"eval_{model_key}.json"
    out.write_text(json.dumps({"report": report, "rows": rows}, indent=2))
    print(json.dumps(report["overall"], indent=2)); print("wrote", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="laion-clap", choices=list(REGISTRY))
    ap.add_argument("--axis", default=None, choices=["T", "R", "O"])
    a = ap.parse_args()
    run(a.model, a.axis)
