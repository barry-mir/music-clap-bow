# Copy to env.sh, edit, then `source env.sh` before running anything.
export MIRB_ROOT="$(pwd)"                       # repo root; data/ and results/ live here
export MIRB_AUDIO=/path/to/mtg-jamendo/audio    # <MIRB_AUDIO>/<jamendo_path> must exist
export MIRB_CLIPS="$MIRB_ROOT/audio/clips_10s"  # written by mirb.prepare_clips
export MIRB_ORDER="$MIRB_ROOT/audio/mirb_order" # written by mirb.build_order
export MOISESDB_ROOT=/path/to/moisesdb_v0.1     # only for mirb.build_order
export LAION_CLAP_CKPT=/path/to/music_audioset_epoch_15_esc_90.14.pt
export CLAMP3_REPO=/path/to/clamp3              # only for --model clamp3
export CLAMP3_ENV=/path/to/clamp3/env
# export QWEN_DEVICE_MAP=auto                   # shard the 7B models across two GPUs
# export HF_HOME=/big/disk/hf                   # Hugging Face cache (~16 GB per 7B model)
