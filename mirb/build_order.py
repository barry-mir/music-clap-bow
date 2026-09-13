"""Build the MIRB-Order audio from MoisesDB stems using data/mirb_order_manifest.csv.

MoisesDB (https://github.com/moises-ai/moises-db) is distributed under its own
licence, so only the recipe is released. For each manifest row the two target
stems (stem_A, stem_B) of one track are taken over the 10 s window starting at
window_start_s, every other stem of the track is added unchanged, and two mixes
are written that differ only in which target instrument enters first:

    A_first.wav : A from the start, B muted at the start (MUTE_SAMPLES)
    B_first.wav : B from the start, A muted at the start (MUTE_SAMPLES)

The mute is a fixed 110,250 samples (4.59 s at 24 kHz) with a 661-sample fade-in;
these sample counts reproduce the released benchmark audio exactly.

Both are loudness-matched and peak-normalised together; caption_A_first is
correct for A_first and caption_B_first for B_first, so the text prior cancels.

    MOISESDB_ROOT=/path/to/moisesdb_v0.1 python -m mirb.build_order
"""
import argparse
import glob
import os

import numpy as np
import soundfile as sf

from mirb.data import load_order_manifest
from mirb.paths import MOISESDB, ORDER_DIR, ORDER_SR

WIN = 10.0            # s
MUTE_SAMPLES = 110250  # the later instrument is muted for this many samples (4.59 s at 24 kHz)
FADE_SAMPLES = 661     # anti-click fade-in at the mute boundary


def load_stem(track_dir, stem, sr):
    """Sum all sub-track wavs of one stem, mono, at sr. None if the stem is absent."""
    import librosa
    files = sorted(glob.glob(os.path.join(track_dir, stem, "*.wav")))
    if not files:
        return None
    mix = None
    for f in files:
        y, _ = librosa.load(f, sr=sr, mono=True)
        if mix is None:
            mix = y
        else:
            n = min(len(mix), len(y)); mix = mix[:n] + y[:n]
    return mix


def front_mask(y):
    z = y.copy()
    z[:MUTE_SAMPLES] = 0.0
    z[MUTE_SAMPLES:MUTE_SAMPLES + FADE_SAMPLES] *= np.linspace(0, 1, FADE_SAMPLES)
    return z


def loudness_match(y, target_rms=0.08):
    r = np.sqrt(np.mean(y ** 2) + 1e-12)
    return y * (target_rms / r)


def build_pair(track_dir, stem_a, stem_b, start_s, sr=ORDER_SR):
    a, b = load_stem(track_dir, stem_a, sr), load_stem(track_dir, stem_b, sr)
    if a is None or b is None:
        raise FileNotFoundError(f"{track_dir}: missing stem {stem_a if a is None else stem_b}")
    s, w = int(round(start_s * sr)), int(WIN * sr)
    seg_a, seg_b = a[s:s + w], b[s:s + w]
    if len(seg_a) < w or len(seg_b) < w:
        raise ValueError(f"{track_dir}: window past the end of a stem")
    others = np.zeros(w, dtype=np.float32)
    present = {d for d in os.listdir(track_dir) if os.path.isdir(os.path.join(track_dir, d))}
    for st in present - {stem_a, stem_b}:
        o = load_stem(track_dir, st, sr)
        if o is None:
            continue
        seg = o[s:s + w]
        others[:len(seg)] += seg
    a_first = loudness_match(others + seg_a + front_mask(seg_b))
    b_first = loudness_match(others + seg_b + front_mask(seg_a))
    peak = max(np.abs(a_first).max(), np.abs(b_first).max(), 1e-6)
    if peak > 1:
        a_first /= peak; b_first /= peak
    return a_first.astype(np.float32), b_first.astype(np.float32)


def main(overwrite=False):
    rows = load_order_manifest(with_audio=False)
    ORDER_DIR.mkdir(parents=True, exist_ok=True)
    done = 0
    for r in rows:
        out = ORDER_DIR / r["pair_id"]
        if (out / "B_first.wav").exists() and not overwrite:
            done += 1; continue
        td = MOISESDB / r["moisesdb_track"]
        if not td.exists():
            raise FileNotFoundError(f"MoisesDB track {td} not found; set MOISESDB_ROOT")
        a_first, b_first = build_pair(str(td), r["stem_A"], r["stem_B"], r["window_start_s"])
        out.mkdir(exist_ok=True)
        sf.write(out / "A_first.wav", a_first, ORDER_SR)
        sf.write(out / "B_first.wav", b_first, ORDER_SR)
        done += 1
        if done % 25 == 0:
            print(f"  {done}/{len(rows)}", flush=True)
    print(f"MIRB-Order ready: {done}/{len(rows)} pairs -> {ORDER_DIR}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()
    main(a.overwrite)
