"""Cut the 10 s benchmark clips from the MTG-Jamendo source tracks.

MIRB does not redistribute audio. Download the full tracks from the MTG-Jamendo
dataset (https://github.com/MTG/mtg-jamendo-dataset, the `audio` files) so that
<MIRB_AUDIO>/<jamendo_path> exists for every row of data/mirb_annotations.csv,
then run

    python -m mirb.prepare_clips

Each annotation gives a window [crop_start_s, crop_end_s] chosen by the annotator
on the full track. The clip is written as <MIRB_CLIPS>/<clip_id>_<axis>.wav,
10 s, 48 kHz, mono, unmodified apart from the cut.
"""
import argparse

import numpy as np
import soundfile as sf

from mirb.data import load_annotations, clip_path
from mirb.paths import AUDIO, CLIPS, CLIP_SECONDS, CLIP_SR


def cut(src, start_s, out_path):
    import librosa
    # identical call to the one used when the benchmark was built (librosa may return 480001 samples)
    y, _ = librosa.load(src, sr=CLIP_SR, mono=True, offset=start_s, duration=CLIP_SECONDS)
    need = int(CLIP_SECONDS * CLIP_SR)
    if len(y) < need:
        y = np.pad(y, (0, need - len(y)))
    sf.write(out_path, y, CLIP_SR)


def main(overwrite=False):
    CLIPS.mkdir(parents=True, exist_ok=True)
    rows = load_annotations()
    missing, done = [], 0
    for r in rows:
        src = AUDIO / r["jamendo_path"]
        dst = clip_path(r["clip_id"], r["axis"])
        if not src.exists():
            missing.append(str(src)); continue
        if dst.exists() and not overwrite:
            done += 1; continue
        cut(src, r["crop_start_s"], dst)
        done += 1
    print(f"clips ready: {done}/{len(rows)} -> {CLIPS}")
    if missing:
        print(f"missing source tracks: {len(missing)} (first: {missing[0]})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()
    main(a.overwrite)
