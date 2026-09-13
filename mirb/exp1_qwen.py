"""Exp. 1, LALM: Qwen2-Audio-7B-Instruct scored generatively.

s(a, c) = P("Yes") normalised over {Yes, No} at the next token, given the audio and
the prompt below. Same pairs, anchors, and output format as exp1_swap.py, so
mirb.stats and mirb.summarize treat both alike (results/eval_qwen2-audio.json).

    python -m mirb.exp1_qwen                    # one 24 GB GPU
    QWEN_DEVICE_MAP=auto python -m mirb.exp1_qwen   # shard the 7B model across GPUs

Note: transformers >= 5 renamed the processor keyword `audios` to `audio` and
silently ignores the old name, so we assert that audio features reach the model.
"""
import json
import os

import numpy as np
import torch
from scipy import stats

from mirb.data import load_pairs
from mirb.paths import RESULTS

MODEL_ID = "Qwen/Qwen2-Audio-7B-Instruct"
PROMPT = ('Does the following description accurately match the music you hear? '
          'Description: "{cap}". Answer with only Yes or No.')


class QwenScorer:
    def __init__(self, device="cuda:0"):
        import librosa
        from transformers import Qwen2AudioForConditionalGeneration, AutoProcessor
        self.librosa = librosa
        self.proc = AutoProcessor.from_pretrained(MODEL_ID)
        dm = os.environ.get("QWEN_DEVICE_MAP", device)
        kw = {"max_memory": {0: "9GiB", 1: "9GiB", "cpu": "30GiB"}} if dm == "auto" else {}
        self.model = Qwen2AudioForConditionalGeneration.from_pretrained(
            MODEL_ID, torch_dtype=torch.float16, device_map=dm, **kw).eval()
        self.device = device if dm != "auto" else next(self.model.parameters()).device
        self.sr = self.proc.feature_extractor.sampling_rate
        self.yes_ids = self._ids(["Yes", " Yes", "yes"])
        self.no_ids = self._ids(["No", " No", "no"])

    def _ids(self, words):
        out = set()
        for w in words:
            t = self.proc.tokenizer(w, add_special_tokens=False).input_ids
            if t:
                out.add(t[0])
        return list(out)

    def p_yes(self, audio_path, caption):
        wav, _ = self.librosa.load(audio_path, sr=self.sr, mono=True)
        conv = [{"role": "user", "content": [{"type": "audio", "audio_url": audio_path},
                                             {"type": "text", "text": PROMPT.format(cap=caption)}]}]
        text = self.proc.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)
        try:
            inputs = self.proc(text=text, audio=[wav], sampling_rate=self.sr, return_tensors="pt")
        except TypeError:  # transformers < 5
            inputs = self.proc(text=text, audios=[wav], sampling_rate=self.sr, return_tensors="pt")
        assert "input_features" in inputs, "processor dropped the audio; check the keyword name"
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            logits = self.model(**inputs).logits[0, -1, :].float()
        probs = torch.softmax(logits, dim=-1)
        py, pn = float(probs[self.yes_ids].sum()), float(probs[self.no_ids].sum())
        return py / (py + pn + 1e-9)


def run():
    pairs = load_pairs()
    sc = QwenScorer()
    rows, n = [], len(pairs)
    for i, r in enumerate(pairs):
        j = (i + n // 2) % n
        while pairs[j]["clip_id"] == r["clip_id"] and n > 1:
            j = (j + 1) % n
        s_pos = sc.p_yes(r["audio_uri"], r["caption_pos"])
        s_neg = sc.p_yes(r["audio_uri"], r["caption_neg"])
        s_nul = sc.p_yes(r["audio_uri"], r["caption_paraphrase"])
        s_wrong = sc.p_yes(r["audio_uri"], pairs[j]["caption_pos"])
        rows.append({"twin_id": r["pair_id"], "clip_id": r["clip_id"], "axis": r["axis"], "style": r["style"],
                     "prior_class": r["prior_class"], "s_pos": s_pos, "s_neg": s_neg, "s_null": s_nul, "s_wrong": s_wrong,
                     "swap_gap": s_pos - s_neg, "null_gap": s_pos - s_nul, "wrong_gap": s_pos - s_wrong})
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{n}", flush=True)
    swap = np.array([x["swap_gap"] for x in rows]); k = int((swap > 0).sum())
    report = {"model": "qwen2-audio-7b-instruct", "n_pairs": n,
              "overall": {"acc": round(k / n, 4), "binom_p": round(stats.binomtest(k, n, 0.5).pvalue, 5),
                          "swap_gap_mean": round(float(swap.mean()), 5)}}
    out = RESULTS / "eval_qwen2-audio.json"
    out.write_text(json.dumps({"report": report, "rows": rows}, indent=2))
    print(json.dumps(report, indent=2)); print("wrote", out)


if __name__ == "__main__":
    run()
