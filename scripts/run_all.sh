#!/usr/bin/env bash
# Run every experiment and produce results/summary.json + figures. Assumes audio is prepared.
set -e
cd "$(dirname "$0")/.."
for m in laion-clap ms-clap muq-mulan clamp3; do
  python -m masb.exp1_swap  --model $m
  python -m masb.exp3_text  --model $m
  python -m masb.exp4_order --model $m
done
python -m masb.exp1_qwen
python -m masb.exp4_order --model qwen2-audio
python -m masb.exp2_floor
python -m masb.stats
python -m masb.summarize
( cd figures && python fig_swapacc.py && python fig_agree.py && python fig_textenc.py )
