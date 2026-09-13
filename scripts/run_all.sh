#!/usr/bin/env bash
# Run every experiment and produce results/summary.json + figures. Assumes audio is prepared.
set -e
cd "$(dirname "$0")/.."
for m in laion-clap ms-clap muq-mulan clamp3; do
  python -m mirb.exp1_swap  --model $m
  python -m mirb.exp3_text  --model $m
  python -m mirb.exp4_order --model $m
done
python -m mirb.exp1_qwen
python -m mirb.exp4_order --model qwen2-audio
python -m mirb.exp2_floor
python -m mirb.stats
python -m mirb.summarize
( cd figures && python fig_swapacc.py && python fig_agree.py && python fig_textenc.py )
