"""Exp. 4, symmetric audio onset-order test on MASB-Order.

Each pair has two mixes of one track that differ only in which instrument enters
first (a+ = A first, a- = B first) and two captions (c+ = "A enters before B",
c- = "B enters before A"). Two comparisons per pair, both scored with the same
s(a, c) as Exp. 1:
    a+ is correct if s(a+, c+) > s(a+, c-)
    a- is correct if s(a-, c-) > s(a-, c+)
A caption prior helps on one comparison and hurts on the other, so it cancels and
chance is exactly 0.5. Reports that accuracy (Wilson 95% CI, exact binomial p),
plus the mirror view (fixed caption, pick the matching mix) and the audio
embedding distances for reference. Writes results/order/order_<model>.json.

    python -m masb.exp4_order --model laion-clap      # ms-clap | muq-mulan | clamp3 | qwen2-audio
"""
import argparse
import json
import math

import numpy as np

from masb.data import load_order_manifest
from masb.models import REGISTRY
from masb.paths import RESULTS


def wilson(k, n, z=1.96):
    p = k / n; d = 1 + z * z / n; c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return [round((c - h) / d, 4), round((c + h) / d, 4)]


def accp(flags):
    from scipy import stats
    k, n = int(sum(flags)), len(flags)
    return {"n": n, "acc": round(k / n, 4), "ci95": wilson(k, n), "binom_p": round(stats.binomtest(k, n, 0.5).pvalue, 6)}


def score_rows(items, s):
    """s(audio_path, caption) -> float. Returns per-pair rows with the four scores."""
    rows = []
    for it in items:
        rows.append({"pair_id": it["pair_id"],
                     "sAA": s(it["A_first"], it["caption_A_first"]), "sAB": s(it["A_first"], it["caption_B_first"]),
                     "sBA": s(it["B_first"], it["caption_A_first"]), "sBB": s(it["B_first"], it["caption_B_first"])})
    return rows


def run(model_key):
    items = load_order_manifest()
    out = {"model": model_key, "n_pairs": len(items)}
    if model_key == "qwen2-audio":
        from masb.exp1_qwen import QwenScorer
        sc = QwenScorer()
        rows = score_rows(items, lambda a, c: float(sc.p_yes(a, c)))
    else:
        model = REGISTRY[model_key]()
        files = sorted({it[k] for it in items for k in ("A_first", "B_first")})
        aemb = dict(zip(files, model.embed_audio(files)))
        caps = sorted({it[k] for it in items for k in ("caption_A_first", "caption_B_first")})
        temb = dict(zip(caps, model.embed_text(caps)))
        rows = score_rows(items, lambda a, c: float(aemb[a] @ temb[c]))
        # audio-embedding distances for reference: order flip vs a different pair's mix
        n = len(items)
        for i, (it, r) in enumerate(zip(items, rows)):
            j = (i + n // 2) % n
            r["d_order"] = 1 - float(aemb[it["A_first"]] @ aemb[it["B_first"]])
            r["d_other"] = 1 - float(aemb[it["A_first"]] @ aemb[items[j]["A_first"]])
        out["audio_distance"] = {"median_d_order": round(float(np.median([r["d_order"] for r in rows])), 4),
                                 "median_d_other": round(float(np.median([r["d_other"] for r in rows])), 4)}
    audio_side = [int(r["sAA"] > r["sAB"]) for r in rows] + [int(r["sBB"] > r["sBA"]) for r in rows]
    text_side = [int(r["sAA"] > r["sBA"]) for r in rows] + [int(r["sBB"] > r["sAB"]) for r in rows]
    out["symmetric"] = accp(audio_side)          # the paper's Table 1
    out["mirror_view"] = accp(text_side)         # fixed caption, pick the matching mix
    out["rows"] = rows
    d = RESULTS / "order"; d.mkdir(exist_ok=True, parents=True)
    (d / f"order_{model_key}.json").write_text(json.dumps(out, indent=2))
    print(json.dumps({k: v for k, v in out.items() if k != "rows"}, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(REGISTRY) + ["qwen2-audio"])
    a = ap.parse_args()
    run(a.model)
