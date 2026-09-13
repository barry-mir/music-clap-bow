"""Loaders for the released benchmark files (CSV). No database needed.

A *pair* is one caption pair (c+, c-) plus its paraphrase for one recording and one
property; each recording has two pairs (strict template, natural rewrite).
"""
import csv
from pathlib import Path

from mirb.paths import ANNOTATIONS_CSV, CAPTIONS_CSV, ORDER_MANIFEST_CSV, CLIPS, ORDER_DIR

AXES = ("T", "R", "O")
AXIS_NAME = {"T": "timbre", "R": "lead versus accompaniment", "O": "onset order"}


def _read(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def load_annotations(axis=None):
    rows = _read(ANNOTATIONS_CSV)
    for r in rows:
        r["crop_start_s"] = float(r["crop_start_s"]); r["crop_end_s"] = float(r["crop_end_s"])
        r["track_duration_s"] = float(r["track_duration_s"]); r["confidence"] = int(r["confidence"])
    return [r for r in rows if axis is None or r["axis"] == axis]


def clip_path(clip_id, axis):
    """The 10 s benchmark clip for (recording, property), written by prepare_clips.py."""
    return CLIPS / f"{clip_id}_{axis}.wav"


def load_pairs(axis=None, with_audio=True):
    """Caption pairs with the fields the experiment scripts use:
    pair_id, clip_id, axis, style, caption_pos, caption_neg, caption_paraphrase, prior_class,
    and audio_uri (path to the clip) when with_audio is True."""
    rows = _read(CAPTIONS_CSV)
    out = []
    for r in rows:
        if axis and r["axis"] != axis:
            continue
        r["pair_id"] = int(r["pair_id"])
        if with_audio:
            p = clip_path(r["clip_id"], r["axis"])
            if not p.exists():
                raise FileNotFoundError(f"missing clip {p}; run `python -m mirb.prepare_clips` first")
            r["audio_uri"] = str(p)
        out.append(r)
    return out


def load_order_manifest(with_audio=True):
    """MIRB-Order pairs: pair_id, moisesdb_track, stem_A, stem_B, name_A, name_B,
    window_start_s, caption_A_first, caption_B_first, and (with_audio) A_first / B_first wav paths."""
    rows = _read(ORDER_MANIFEST_CSV)
    for r in rows:
        r["window_start_s"] = float(r["window_start_s"])
        if with_audio:
            d = ORDER_DIR / r["pair_id"]
            r["A_first"] = str(d / "A_first.wav"); r["B_first"] = str(d / "B_first.wav")
            if not Path(r["A_first"]).exists():
                raise FileNotFoundError(f"missing {r['A_first']}; run `python -m mirb.build_order` first")
    return rows
