# MASB data files

All three files are UTF-8 CSV with a header row. Instrument and timbre words are lower-case English
from the closed vocabularies in `masb/vocab.py` (plus a few annotator-added words).

## `masb_annotations.csv` (400 recordings)

One row per (recording, property). A recording is annotated for exactly one property.

| column | meaning |
|---|---|
| `clip_id` | MTG-Jamendo track id (also the Song Describer clip id) |
| `source` | `SDD`: the track is in the Song Describer Dataset |
| `jamendo_path` | path of the full track inside the MTG-Jamendo audio release, e.g. `74/1009674.mp3` |
| `track_duration_s` | duration of that full track |
| `axis` | property: `T` timbre, `R` lead versus accompaniment, `O` onset order |
| `crop_start_s`, `crop_end_s` | the 10 s window on the full track chosen by the annotator where the property is clearest |
| `inst_1`, `inst_2` | the two instruments. `T`: instrument carrying `timbre_1` / `timbre_2`. `R`: `inst_1` plays the melody, `inst_2` accompanies. `O`: `inst_1` enters before `inst_2` |
| `timbre_1`, `timbre_2` | timbre words for property `T`, empty otherwise |
| `confidence` | annotator confidence, 1 to 5 |

The benchmark clip for a row is `<MASB_CLIPS>/<clip_id>_<axis>.wav`: the window cut from the full
track, 10 s, 48 kHz, mono, no other processing (`python -m masb.prepare_clips`). Windows can start
later than 120 s, so the full MTG-Jamendo tracks are needed, not the 2 min Song Describer excerpts.

## `masb_captions.csv` (800 caption pairs)

Two rows per recording, `style = strict` (fixed template) and `style = natural` (rewrite of the
same content). Generated from the annotations by `python -m masb.build_captions`.

| column | meaning |
|---|---|
| `pair_id` | integer id used in all result files (`twin_id`) |
| `clip_id`, `axis` | link to the annotation |
| `style` | `strict` or `natural` |
| `caption_pos` | `c+`, the caption that matches the recording |
| `caption_neg` | `c-`, the attribute swap: same words, the two instruments' slots exchanged |
| `caption_paraphrase` | same meaning as `c+`, reworded through a fixed synonym map (no LLM) |
| `prior_class` | whether `c+` is the more frequent assignment in the Song Describer caption corpus: `consistent`, `violating`, or `unknown` (always `unknown` for `O`) |

Templates (`strict`):

```
T:  a {timbre_1} {inst_1} and a {timbre_2} {inst_2}
R:  the {inst_1} plays the melody while the {inst_2} accompanies
O:  the {inst_1} enters before the {inst_2}
```

## `masb_order_manifest.csv` (300 MASB-Order pairs)

| column | meaning |
|---|---|
| `pair_id` | `i0000` to `i0299` |
| `moisesdb_track` | MoisesDB v0.1 track directory name (UUID) |
| `stem_A`, `stem_B` | the two MoisesDB stem folders used as target instruments |
| `name_A`, `name_B` | their caption names (derived from the stem's track type) |
| `window_start_s` | start of the 10 s window on the stems |
| `caption_A_first` | `c+` for the mix in which A enters first ("the {A} enters before the {B}") |
| `caption_B_first` | `c-` for that mix, and `c+` for the mix in which B enters first |

`python -m masb.build_order` writes `<MASB_ORDER>/<pair_id>/A_first.wav` and `B_first.wav`: 10 s,
24 kHz, mono; the target stems plus every other stem of the track over the window, with the later
instrument muted for the first 5 s, loudness-matched and peak-normalised as a pair.
`window_start_s` lies on a 0.459375 s grid (the activity-frame size used when the windows were chosen).

## License

The annotations, captions, and manifest are released under CC BY 4.0 (see `LICENSE` in this
directory). The audio is not included; it keeps the licences of MTG-Jamendo (per-track Creative
Commons, see its `audio_licenses`) and MoisesDB.
