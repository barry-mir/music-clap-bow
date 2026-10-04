"""Statistical enhancement over the raw eval JSONs.

Per model x axis and overall:
  - swap accuracy, n, Wilson 95% CI, exact two-sided binomial p vs 0.5
  - Holm-Bonferroni corrected p across the whole model x axis family
  - paired Wilcoxon signed-rank (swap_gap vs null_gap) + rank-biserial effect size
  - mean swap/null/wrong gaps with 10k-bootstrap 95% CI
Text encoders (Exp. 3):
  - swap>paraphrase accuracy, Wilson CI, binomial p; Wilcoxon(d_swap vs d_paraphrase)
Floor: per-axis accuracy, CI, binomial p.
Listening margin: audio acc - floor acc per model x axis, bootstrap CI.

  python -m masb.stats   ->  results/stats.json
"""
import glob
import json
import math
from collections import defaultdict

import numpy as np
from scipy import stats

from masb.paths import RESULTS

AXES = ["T", "R", "O"]
RNG = np.random.default_rng(0)  # fixed seed: reproducible bootstrap


def wilson(k, n, z=1.96):
    if n == 0:
        return (None, None)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return round((c - h) / d, 4), round((c + h) / d, 4)


def binom_p(k, n):
    return stats.binomtest(k, n, 0.5).pvalue if n else 1.0


def boot_ci(x, fn=np.mean, B=10000):
    x = np.asarray(x, float)
    if len(x) == 0:
        return (None, None)
    idx = RNG.integers(0, len(x), size=(B, len(x)))
    vals = fn(x[idx], axis=1)
    return round(float(np.percentile(vals, 2.5)), 5), round(float(np.percentile(vals, 97.5)), 5)


def rank_biserial_wilcoxon(a, b):
    """Paired Wilcoxon on a-b, plus matched-pairs rank-biserial effect size."""
    d = np.asarray(a, float) - np.asarray(b, float)
    d = d[d != 0]
    if len(d) < 1:
        return {"p": float("nan"), "rank_biserial": float("nan"), "n": 0}
    try:
        w, p = stats.wilcoxon(d)
    except ValueError:
        return {"p": float("nan"), "rank_biserial": float("nan"), "n": len(d)}
    ranks = stats.rankdata(np.abs(d))
    rpos = ranks[d > 0].sum()
    rneg = ranks[d < 0].sum()
    tot = rpos + rneg
    rb = float((rpos - rneg) / tot) if tot else 0.0
    return {"p": round(float(p), 6), "rank_biserial": round(rb, 4), "n": int(len(d))}


def holm(pvals):
    """Holm-Bonferroni. pvals: dict key->p. Returns dict key->corrected p."""
    items = sorted(((k, v) for k, v in pvals.items() if v is not None), key=lambda x: x[1])
    m = len(items)
    out = {}
    prev = 0.0
    for i, (k, p) in enumerate(items):
        adj = min(1.0, (m - i) * p)
        adj = max(adj, prev)  # enforce monotonicity
        out[k] = round(adj, 6)
        prev = adj
    return out


def acc_block(flags, gaps=None):
    n = len(flags)
    k = int(np.sum(flags))
    lo, hi = wilson(k, n)
    b = {"n": n, "acc": round(k / n, 4) if n else None, "ci95": [lo, hi], "binom_p": round(binom_p(k, n), 6)}
    return b


def load_rows():
    audio, text, floor = {}, {}, None
    for f in sorted(glob.glob(str(RESULTS / "eval_*.json"))):
        d = json.loads(open(f).read())
        audio[d["report"]["model"]] = d["rows"]
    for f in sorted(glob.glob(str(RESULTS / "text_text_*.json"))):
        d = json.loads(open(f).read())
        text[d["report"]["model"]] = d["rows"]
    fp = RESULTS / "floor_lalm.json"
    if not fp.exists():
        fp = RESULTS / "floor.json"
    if fp.exists():
        floor = json.loads(fp.read_text())["rows"]
    return audio, text, floor


def main():
    audio, text, floor = load_rows()
    out = {"audio": {}, "text_encoder": {}, "floor": {}, "listening_margin": {}}

    # ---- floor per axis (needed for margins) ----
    floor_acc = {}
    if floor:
        for ax in AXES + ["ALL"]:
            rs = floor if ax == "ALL" else [r for r in floor if r["axis"] == ax]
            b = acc_block([r["correct"] for r in rs])
            out["floor"][ax] = b
            floor_acc[ax] = b["acc"]

    # ---- audio models: swap accuracy + gap tests ----
    family_p = {}  # (model,axis) -> binom p, for Holm across the whole grid
    for m, rows in audio.items():
        out["audio"][m] = {}
        byax = defaultdict(list)
        for r in rows:
            byax[r["axis"]].append(r)
        for ax in AXES + ["ALL"]:
            rs = rows if ax == "ALL" else byax[ax]
            blk = acc_block([r["swap_gap"] > 0 for r in rs])
            blk["wilcoxon_swap_vs_null"] = rank_biserial_wilcoxon(
                [r["swap_gap"] for r in rs], [r["null_gap"] for r in rs])
            blk["swap_gap_mean"] = round(float(np.mean([r["swap_gap"] for r in rs])), 5)
            blk["swap_gap_ci95"] = boot_ci([r["swap_gap"] for r in rs])
            blk["null_gap_mean"] = round(float(np.mean([r["null_gap"] for r in rs])), 5)
            blk["wrong_gap_mean"] = round(float(np.mean([r["wrong_gap"] for r in rs])), 5)
            out["audio"][m][ax] = blk
            if ax != "ALL":
                family_p[f"{m}|{ax}"] = blk["binom_p"]
            # listening margin vs floor
            if ax in floor_acc and floor_acc[ax] is not None and blk["acc"] is not None:
                out["listening_margin"].setdefault(m, {})[ax] = round(blk["acc"] - floor_acc[ax], 4)
    out["holm_swap_acc_vs_chance"] = holm(family_p)

    # ---- text encoders (Exp. 3) ----
    for m, rows in text.items():
        out["text_encoder"][m] = {}
        byax = defaultdict(list)
        for r in rows:
            byax[r["axis"]].append(r)
        for ax in AXES + ["ALL"]:
            rs = rows if ax == "ALL" else byax[ax]
            blk = acc_block([r["text_binding_correct"] for r in rs])
            blk["wilcoxon_dswap_vs_dpar"] = rank_biserial_wilcoxon(
                [r["d_swap"] for r in rs], [r["d_paraphrase"] for r in rs])
            blk["mean_d_swap"] = round(float(np.mean([r["d_swap"] for r in rs])), 5)
            blk["mean_d_paraphrase"] = round(float(np.mean([r["d_paraphrase"] for r in rs])), 5)
            blk["mean_d_other"] = round(float(np.mean([r["d_other"] for r in rs])), 5)
            out["text_encoder"][m][ax] = blk

    (RESULTS / "stats.json").write_text(json.dumps(out, indent=2))

    # ---- console summary ----
    print("=== SWAP ACCURACY vs chance (Wilson 95% CI, binom p, Holm-corrected) ===")
    for m in out["audio"]:
        a = out["audio"][m]["ALL"]
        hp = out["holm_swap_acc_vs_chance"]
        # ALL not in family; show axis Holm range
        print(f"{m:24s} ALL acc={a['acc']} CI{a['ci95']} p={a['binom_p']}")
        for ax in AXES:
            b = out["audio"][m][ax]
            print(f"   {ax}: acc={b['acc']} CI{b['ci95']} p={b['binom_p']} "
                  f"holm={hp.get(m+'|'+ax)} | Wilcoxon(swap>null) p={b['wilcoxon_swap_vs_null']['p']} "
                  f"rb={b['wilcoxon_swap_vs_null']['rank_biserial']}")
    print("\n=== TEXT ENCODER (Exp. 3): swap>paraphrase vs chance ===")
    for m in out["text_encoder"]:
        b = out["text_encoder"][m]["ALL"]
        print(f"{m:22s} acc={b['acc']} CI{b['ci95']} p={b['binom_p']} "
              f"Wilcoxon(dswap>dpar) p={b['wilcoxon_dswap_vs_dpar']['p']} rb={b['wilcoxon_dswap_vs_dpar']['rank_biserial']}")
    print("\n=== FLOOR ===")
    for ax in AXES + ["ALL"]:
        b = out["floor"].get(ax, {})
        print(f"   {ax}: acc={b.get('acc')} CI{b.get('ci95')} p={b.get('binom_p')}")
    print("\n=== LISTENING MARGIN (audio acc - floor acc) ===")
    for m in out["listening_margin"]:
        print(f"{m:24s} " + " ".join(f"{ax}={out['listening_margin'][m][ax]:+.3f}" for ax in AXES))
    print("\nwrote", RESULTS / "stats.json")


if __name__ == "__main__":
    main()
