"""Central paths. Everything is relative to MASB_ROOT (default: the repo directory).

Override with environment variables:
  MASB_ROOT        repo root (data/, results/ live here)
  MASB_AUDIO       directory with the MTG-Jamendo source tracks (<MASB_AUDIO>/<jamendo_path>)
  MASB_CLIPS       directory for the 10 s benchmark clips written by prepare_clips.py
  MASB_ORDER       directory for the MASB-Order mixes written by build_order.py
  MOISESDB_ROOT    MoisesDB v0.1 root (needed only to build MASB-Order audio)
  LAION_CLAP_CKPT  path to music_audioset_epoch_15_esc_90.14.pt (LAION-CLAP music checkpoint)
  CLAMP3_REPO / CLAMP3_ENV  CLaMP 3 checkout and its python env (see README)
"""
import os
from pathlib import Path

ROOT = Path(os.environ.get("MASB_ROOT", Path(__file__).resolve().parent.parent))
DATA = ROOT / "data"
RESULTS = Path(os.environ.get("MASB_RESULTS", ROOT / "results"))

ANNOTATIONS_CSV = DATA / "masb_annotations.csv"
CAPTIONS_CSV = DATA / "masb_captions.csv"
ORDER_MANIFEST_CSV = DATA / "masb_order_manifest.csv"

AUDIO = Path(os.environ.get("MASB_AUDIO", ROOT / "audio" / "jamendo"))
CLIPS = Path(os.environ.get("MASB_CLIPS", ROOT / "audio" / "clips_10s"))
ORDER_DIR = Path(os.environ.get("MASB_ORDER", ROOT / "audio" / "masb_order"))
MOISESDB = Path(os.environ.get("MOISESDB_ROOT", ROOT / "audio" / "moisesdb_v0.1"))

# Benchmark clip format: 10 s, 48 kHz mono. Every model resamples from here.
CLIP_SECONDS = 10.0
CLIP_SR = 48000
# MASB-Order mixes: 10 s, 24 kHz mono, second instrument muted for the first 5 s.
ORDER_SR = 24000

LAION_CLAP_CKPT = Path(os.environ.get("LAION_CLAP_CKPT", ROOT / "checkpoints" / "music_audioset_epoch_15_esc_90.14.pt"))

RESULTS.mkdir(parents=True, exist_ok=True)
