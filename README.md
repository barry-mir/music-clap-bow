# MASB: Music Attribute-Swap Benchmark

Code and data for **Don't CLAP: Are Music-Text Models Bag-of-Words?** (Cheng and Lerch, submitted to ICASSP 2027).

The benchmark tests whether a music-text model reads which attribute belongs to which instrument in a caption: which
instrument carries which timbre, which one leads and which accompanies, and which enters first.
For each recording we build an original caption `c+` and an **attribute swap** `c-` that contain the
same words with one property exchanged between the two instruments, and ask whether the model
scores the audio higher against `c+` than against `c-`.

```
a distorted guitar and a clean piano      c+  (matches the recording)
a clean guitar and a distorted piano      c-  (same words, property swapped)
```

Four contrastive music-text models (LAION-CLAP, MS-CLAP 2023, MuQ-MuLan, CLaMP 3) rank `c+` above
`c-` at chance, and their text embeddings barely move under the swap. The audio-language model
Qwen2-Audio does better, but its advantage rests largely on a language prior: running the same
model without audio already reproduces most of its choices.

## What is released

| file | rows | content |
|---|---|---|
| `data/masb_annotations.csv` | 400 | one recording per row: MTG-Jamendo track path, the 10 s window, the two instruments and (for timbre) their timbre words, annotator confidence |
| `data/masb_captions.csv` | 800 | two caption pairs per recording (strict template and natural rewrite): `c+`, `c-`, a paraphrase, and the corpus prior class |
| `data/masb_order_manifest.csv` | 300 | MASB-Order: MoisesDB track, the two stems, the 10 s window, and the two captions for the symmetric onset-order test |

Audio is **not** redistributed. MASB recordings are unmodified excerpts of Creative-Commons tracks from
the [Song Describer Dataset](https://github.com/mulab-mir/song-describer-dataset) /
[MTG-Jamendo](https://github.com/MTG/mtg-jamendo-dataset); MASB-Order mixes are built from
[MoisesDB](https://github.com/moises-ai/moises-db) stems. Both are rebuilt deterministically from the
CSVs with the scripts below. See `data/README.md` for the column definitions.

## Setup

```bash
pip install -r requirements.txt
cp env.example.sh env.sh   # edit paths, then: source env.sh
```

Model dependencies (install what you need):

- **LAION-CLAP**: `pip install laion_clap`; download the music checkpoint
  `music_audioset_epoch_15_esc_90.14.pt` from the LAION-CLAP release and point `LAION_CLAP_CKPT` at it.
- **MS-CLAP 2023**: `pip install msclap` (weights download on first use).
- **MuQ-MuLan**: `pip install muq` (pulls `OpenMuQ/MuQ-MuLan-large` from Hugging Face).
- **CLaMP 3**: clone https://github.com/sanderwood/clamp3, install its requirements in a separate
  environment (its pinned `transformers` conflicts with Qwen2-Audio), download the SaaS checkpoint per
  its README, and set `CLAMP3_REPO` and `CLAMP3_ENV`.
- **Qwen2-Audio-7B-Instruct** (Exp. 1 and Exp. 2) and optionally **Qwen2.5-7B-Instruct** (the LLM
  reference for Exp. 2): `transformers>=4.45` and about 16 GB of GPU memory each; set
  `QWEN_DEVICE_MAP=auto` to shard across two smaller GPUs.

## Reproduce the paper

```bash
# 1. audio
python -m masb.prepare_clips        # cut the 400 clips from <MASB_AUDIO>/<jamendo_path>
python -m masb.build_order          # build the 300 MASB-Order pairs from <MOISESDB_ROOT>

# 2. experiments
for m in laion-clap ms-clap muq-mulan clamp3; do
  python -m masb.exp1_swap  --model $m   # Exp. 1  CLAP score under the attribute swap
  python -m masb.exp3_text  --model $m   # Exp. 3  text embedding only
  python -m masb.exp4_order --model $m   # Exp. 4  symmetric onset-order test
done
python -m masb.exp1_qwen                 # Exp. 1  Qwen2-Audio (generative scoring)
python -m masb.exp4_order --model qwen2-audio
python -m masb.exp2_floor                # Exp. 2  the LALM without audio (its own text prior)

# 3. numbers and figures
python -m masb.stats                     # Wilson CIs, exact binomial p, Holm correction
python -m masb.summarize                 # kappa, prior-stratified accuracy, medians -> results/summary.json
cd figures && python fig_swapacc.py && python fig_agree.py && python fig_textenc.py
```

`scripts/run_all.sh` chains steps 2 and 3. Every script writes JSON with per-pair raw scores under
`results/`, so any number in the paper can be recomputed from them.

## Layout

```
masb/
  data.py           CSV loaders (pairs, annotations, MASB-Order manifest)
  paths.py          all paths, overridable by environment variables
  prepare_clips.py  cut the benchmark clips from MTG-Jamendo tracks
  build_captions.py regenerate captions.csv from annotations.csv (templates, swap, paraphrase map)
  build_order.py    build MASB-Order mixes from MoisesDB stems
  models.py         LAION-CLAP / MS-CLAP / MuQ-MuLan wrappers (embed_audio, embed_text)
  clamp3_model.py   CLaMP 3 wrapper (subprocess into its own environment)
  exp1_swap.py      Exp. 1, contrastive models
  exp1_qwen.py      Exp. 1, Qwen2-Audio
  exp2_floor.py     Exp. 2, text-only floor (--model lalm, the paper's; --model llm reference)
  exp3_text.py      Exp. 3, text-embedding test
  exp4_order.py     Exp. 4, symmetric audio test
  stats.py          confidence intervals, tests, Holm correction
  summarize.py      all paper numbers -> results/summary.json
  vocab.py          instrument list and suggested timbre adjectives used in annotation
figures/            paper figures from results/
data/               the benchmark files
```

## Citation

```bibtex
@inproceedings{cheng2027dontclap,
  title     = {Don't {CLAP}: Are Music-Text Models Bag-of-Words?},
  author    = {Cheng, Yuan-Chiao and Lerch, Alexander},
  booktitle = {Proc. IEEE Int. Conf. Acoust., Speech, Signal Process. (ICASSP)},
  year      = {2027}
}
```

## License

Code: MIT (`LICENSE`). Annotations and captions in `data/`: CC BY 4.0 (`data/LICENSE`). The source
audio keeps the licences of MTG-Jamendo (per-track Creative Commons) and MoisesDB (research licence).
