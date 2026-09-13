"""Central paths. Everything is relative to MIRB_ROOT (default: the repo directory).

Override with environment variables:
  MIRB_ROOT        repo root (data/, results/ live here)
  MIRB_AUDIO       directory with the MTG-Jamendo source tracks (<MIRB_AUDIO>/<jamendo_path>)
  MIRB_CLIPS       directory for the 10 s benchmark clips written by prepare_clips.py
  MIRB_ORDER       directory for the MIRB-Order mixes written by build_order.py
  MOISESDB_ROOT    MoisesDB v0.1 root (needed only to build MIRB-Order audio)
  LAION_CLAP_CKPT  path to music_audioset_epoch_15_esc_90.14.pt (LAION-CLAP music checkpoint)
  CLAMP3_REPO / CLAMP3_ENV  CLaMP 3 checkout and its python env (see README)
"""
import os
from pathlib import Path

ROOT = Path(os.environ.get("MIRB_ROOT", Path(__file__).resolve().parent.parent))
DATA = ROOT / "data"
RESULTS = Path(os.environ.get("MIRB_RESULTS", ROOT / "results"))

ANNOTATIONS_CSV = DATA / "mirb_annotations.csv"
CAPTIONS_CSV = DATA / "mirb_captions.csv"
ORDER_MANIFEST_CSV = DATA / "mirb_order_manifest.csv"

AUDIO = Path(os.environ.get("MIRB_AUDIO", ROOT / "audio" / "jamendo"))
CLIPS = Path(os.environ.get("MIRB_CLIPS", ROOT / "audio" / "clips_10s"))
ORDER_DIR = Path(os.environ.get("MIRB_ORDER", ROOT / "audio" / "mirb_order"))
MOISESDB = Path(os.environ.get("MOISESDB_ROOT", ROOT / "audio" / "moisesdb_v0.1"))

# Benchmark clip format: 10 s, 48 kHz mono. Every model resamples from here.
CLIP_SECONDS = 10.0
CLIP_SR = 48000
# MIRB-Order mixes: 10 s, 24 kHz mono, second instrument muted for the first 5 s.
ORDER_SR = 24000

LAION_CLAP_CKPT = Path(os.environ.get("LAION_CLAP_CKPT", ROOT / "checkpoints" / "music_audioset_epoch_15_esc_90.14.pt"))

RESULTS.mkdir(parents=True, exist_ok=True)
