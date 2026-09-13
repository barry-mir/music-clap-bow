"""Exp. 3, single-modality text test: does the text embedding separate c+ from c-?

No audio. For each pair the text encoder embeds c+ and we measure the cosine
distance d(x, y) = 1 - cos(x, y) from c+ to
    the swapped caption c-      (same words, meaning changed)
    the paraphrase              (different words, meaning kept)
    another recording's c+      (different content, upper reference)
An encoder that keeps instrument relations should move farther under the swap
than under the paraphrase. Writes results/text_text_<model>.json (per-pair
distances); masb.summarize reports medians and the swap > paraphrase rate.

    python -m masb.exp3_text --model laion-clap      # ms-clap | muq-mulan | clamp3
"""
import argparse
import json

import numpy as np
from scipy import stats

from masb.data import load_pairs
from masb.models import REGISTRY
from masb.paths import RESULTS


def run(model_key, axis=None):
    pairs = load_pairs(axis, with_audio=False)
    model = REGISTRY[model_key]()
    E_pos = model.embed_text([t["caption_pos"] for t in pairs])
    E_neg = model.embed_text([t["caption_neg"] for t in pairs])
    E_par = model.embed_text([t["caption_paraphrase"] for t in pairs])
    rows, n = [], len(pairs)
    for i, t in enumerate(pairs):
        j = (i + n // 2) % n
        while pairs[j]["clip_id"] == t["clip_id"] and n > 1:
            j = (j + 1) % n
        d_swap = 1.0 - float(E_pos[i] @ E_neg[i])
        d_par = 1.0 - float(E_pos[i] @ E_par[i])
        d_other = 1.0 - float(E_pos[i] @ E_pos[j])
        rows.append({"twin_id": t["pair_id"], "axis": t["axis"], "style": t["style"],
                     "d_swap": d_swap, "d_paraphrase": d_par, "d_other": d_other,
                     "text_binding_correct": int(d_swap > d_par)})

    def summ(sub):
        k = sum(r["text_binding_correct"] for r in sub)
        return {"n": len(sub), "swap_gt_paraphrase_acc": round(k / len(sub), 4),
                "binom_p": round(stats.binomtest(k, len(sub), 0.5).pvalue, 5),
                "median_d_swap": round(float(np.median([r["d_swap"] for r in sub])), 4),
                "median_d_paraphrase": round(float(np.median([r["d_paraphrase"] for r in sub])), 4),
                "median_d_other": round(float(np.median([r["d_other"] for r in sub])), 4)}

    report = {"model": model.name, "overall": summ(rows),
              "by_axis": {ax: summ([r for r in rows if r["axis"] == ax]) for ax in sorted({r["axis"] for r in rows})}}
    out = RESULTS / f"text_text_{model_key}.json"
    out.write_text(json.dumps({"report": report, "rows": rows}, indent=2))
    print(json.dumps(report["overall"], indent=2)); print("wrote", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="laion-clap", choices=list(REGISTRY))
    ap.add_argument("--axis", default=None, choices=["T", "R", "O"])
    a = ap.parse_args()
    run(a.model, a.axis)
