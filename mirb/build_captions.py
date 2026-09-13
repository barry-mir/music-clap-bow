"""Regenerate data/mirb_captions.csv from data/mirb_annotations.csv.

The released captions.csv is the file the paper used; this script documents how
it was produced and lets you extend the benchmark with new annotations. For each
annotation it writes two caption pairs (strict template, natural rewrite), each
with an original c+, an attribute swap c- (same words, the two instruments'
slots exchanged) and a paraphrase (same meaning, reworded through a fixed
synonym map, no LLM involved). It checks that c+ and c- have the same lemma
multiset.

prior_class (consistent / violating / unknown) records whether the original
assignment is the more frequent one in the Song Describer caption corpus; it is
kept from the released file unless --sdd points at song_describer.csv.

    python -m mirb.build_captions [--sdd path/to/song_describer.csv]
"""
import argparse
import csv
import re
from collections import Counter, defaultdict

from mirb.data import load_annotations, _read
from mirb.paths import CAPTIONS_CSV
from mirb.vocab import find_instruments, TIMBRES

# paraphrase synonym map for timbre words (used only in the paraphrase caption)
TIMBRE_SYN = {
    "distorted": "overdriven", "clean": "cleanly played", "bright": "trebly",
    "dark": "muffled-toned", "muted": "damped", "warm": "rounded",
    "crisp": "sharp-edged", "mellow": "gentle-toned", "harsh": "abrasive",
    "soft": "softly played", "punchy": "percussive", "smooth": "silky",
    "ambient": "atmospheric", "bouncy": "springy", "gritty": "grainy",
}


def captions_T(i1, i2, t1, t2):
    p1, p2 = TIMBRE_SYN.get(t1, t1), TIMBRE_SYN.get(t2, t2)
    return {
        "strict": (f"a {t1} {i1} and a {t2} {i2}",
                   f"a {t2} {i1} and a {t1} {i2}",
                   f"an {p1} {i1} over a {p2} {i2}"),
        "natural": (f"the {i1} sounds {t1} while the {i2} is {t2}",
                    f"the {i1} sounds {t2} while the {i2} is {t1}",
                    f"the {i1} comes across as {p1}, the {i2} as {p2}"),
    }


def captions_R(lead, acc):
    return {
        "strict": (f"the {lead} plays the melody while the {acc} accompanies",
                   f"the {acc} plays the melody while the {lead} accompanies",
                   f"the melody is carried by the {lead}, with the {acc} providing accompaniment"),
        "natural": (f"the {lead} leads with the tune as the {acc} backs it",
                    f"the {acc} leads with the tune as the {lead} backs it",
                    f"the {lead} takes the lead line and the {acc} plays the supporting part"),
    }


def captions_O(first, second):
    return {
        "strict": (f"the {first} enters before the {second}",
                   f"the {second} enters before the {first}",
                   f"the {first} comes in earlier than the {second}"),
        "natural": (f"first the {first} comes in, then the {second}",
                    f"first the {second} comes in, then the {first}",
                    f"the {first} starts first and the {second} joins afterward"),
    }


def build(r):
    if r["axis"] == "T":
        return captions_T(r["inst_1"], r["inst_2"], r["timbre_1"], r["timbre_2"])
    if r["axis"] == "R":
        return captions_R(r["inst_1"], r["inst_2"])
    return captions_O(r["inst_1"], r["inst_2"])


def lemmas(s):
    return Counter(t.rstrip("s") for t in re.findall(r"[a-z]+", s.lower()))


# ---- corpus co-occurrence prior (optional, needs the Song Describer caption csv) ----
LEAD_WORDS = {"melody", "lead", "solo", "solos", "soloing", "melodic", "leads"}
ACC_WORDS = {"accompan", "backing", "rhythm", "supporting", "chords", "comping"}


def corpus_stats(sdd_csv):
    timbre, role = defaultdict(Counter), defaultdict(Counter)
    with open(sdd_csv) as f:
        for row in csv.DictReader(f):
            cap = row["caption"].lower()
            toks = re.findall(r"[a-z]+", cap)
            insts = find_instruments(cap)
            for i, t in enumerate(toks):
                if t in TIMBRES:
                    win = " ".join(toks[i:i + 5])
                    for inst in insts:
                        if inst.split()[-1] in win:
                            timbre[inst][t] += 1
            has_lead = any(w in cap for w in LEAD_WORDS)
            has_acc = any(a in cap for a in ACC_WORDS)
            for inst in insts:
                if has_lead:
                    role[inst]["lead"] += 1
                if has_acc:
                    role[inst]["accomp"] += 1
    return timbre, role


def prior_class(r, timbre, role):
    if r["axis"] == "T":
        self_s = timbre[r["inst_1"]][r["timbre_1"]] + timbre[r["inst_2"]][r["timbre_2"]]
        swap_s = timbre[r["inst_1"]][r["timbre_2"]] + timbre[r["inst_2"]][r["timbre_1"]]
    elif r["axis"] == "R":
        self_s = role[r["inst_1"]]["lead"] + role[r["inst_2"]]["accomp"]
        swap_s = role[r["inst_2"]]["lead"] + role[r["inst_1"]]["accomp"]
    else:
        return "unknown"
    if self_s == swap_s:
        return "unknown"
    return "consistent" if self_s > swap_s else "violating"


def main(sdd_csv=None, out=CAPTIONS_CSV):
    ann = load_annotations()
    old_prior, old_id = {}, {}
    if out.exists():  # keep released pair_ids and prior classes stable
        for c in _read(out):
            old_prior[(c["clip_id"], c["axis"])] = c["prior_class"]
            old_id[(c["clip_id"], c["axis"], c["style"])] = int(c["pair_id"])
    stats = corpus_stats(sdd_csv) if sdd_csv else None
    rows, next_id = [], (max(old_id.values()) + 1 if old_id else 1)
    for r in ann:
        pc = prior_class(r, *stats) if stats else old_prior.get((r["clip_id"], r["axis"]), "unknown")
        for style, (pos, neg, par) in build(r).items():
            assert lemmas(pos) == lemmas(neg), (r["clip_id"], r["axis"], pos, neg)
            pid = old_id.get((r["clip_id"], r["axis"], style))
            if pid is None:
                pid, next_id = next_id, next_id + 1
            rows.append(dict(pair_id=pid, clip_id=r["clip_id"], axis=r["axis"], style=style,
                             caption_pos=pos, caption_neg=neg, caption_paraphrase=par, prior_class=pc))
    rows.sort(key=lambda x: (x["axis"], x["clip_id"], x["style"]))
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print(f"wrote {len(rows)} caption pairs -> {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sdd", default=None, help="song_describer.csv, to recompute prior_class")
    a = ap.parse_args()
    main(a.sdd)
