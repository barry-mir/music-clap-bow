# Copy to env.sh, edit, then `source env.sh` before running anything.
export MASB_ROOT="$(pwd)"                       # repo root; data/ and results/ live here
export MASB_AUDIO=/path/to/mtg-jamendo/audio    # <MASB_AUDIO>/<jamendo_path> must exist
export MASB_CLIPS="$MASB_ROOT/audio/clips_10s"  # written by masb.prepare_clips
export MASB_ORDER="$MASB_ROOT/audio/masb_order" # written by masb.build_order
export MOISESDB_ROOT=/path/to/moisesdb_v0.1     # only for masb.build_order
export LAION_CLAP_CKPT=/path/to/music_audioset_epoch_15_esc_90.14.pt
export CLAMP3_REPO=/path/to/clamp3              # only for --model clamp3
export CLAMP3_ENV=/path/to/clamp3/env
# export QWEN_DEVICE_MAP=auto                   # shard the 7B models across two GPUs
# export HF_HOME=/big/disk/hf                   # Hugging Face cache (~16 GB per 7B model)
