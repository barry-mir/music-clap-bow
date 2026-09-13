"""CLaMP 3 embedding interface.

CLaMP 3 (https://github.com/sanderwood/clamp3) ships as file-based scripts:
`clamp3_embd.py <in_dir> <out_dir> --get_global` detects the modality of each
file (.wav -> MERT features -> audio encoder, .txt -> text encoder) and writes one
(1, 768) global embedding per input. Its pinned transformers version conflicts
with Qwen2-Audio, so we run it in its own environment via subprocess.

Set CLAMP3_REPO (checkout with the SaaS checkpoint downloaded per its README)
and CLAMP3_ENV (a python env with its requirements).
"""
import os
import shutil
import subprocess
import tempfile

import numpy as np

CLAMP3_REPO = os.environ.get("CLAMP3_REPO", "")
CLAMP3_ENV = os.environ.get("CLAMP3_ENV", "")


def _l2(x):
    x = np.asarray(x, dtype=np.float32)
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-8)


class Clamp3:
    name = "clamp3-saas"

    def __init__(self):
        self.py = os.path.join(CLAMP3_ENV, "bin", "python")
        if not (CLAMP3_REPO and CLAMP3_ENV and os.path.exists(self.py)):
            raise RuntimeError("set CLAMP3_REPO and CLAMP3_ENV (see clamp3_model.py docstring)")
        self.env = dict(os.environ)
        self.env["PATH"] = os.path.join(CLAMP3_ENV, "bin") + ":" + self.env.get("PATH", "")
        self.env["PYTHONNOUSERSITE"] = "1"

    def _run(self, items, suffix, writer):
        work = tempfile.mkdtemp(prefix="clamp3_")
        indir, outdir = os.path.join(work, "in"), os.path.join(work, "out")
        os.makedirs(indir)
        stems = []
        for i, it in enumerate(items):
            stem = f"{i:05d}"
            writer(it, os.path.join(indir, stem + suffix))
            stems.append(stem)
        try:
            r = subprocess.run(
                [self.py, "clamp3_embd.py", indir, outdir, "--get_global"],
                cwd=CLAMP3_REPO, env=self.env, capture_output=True, text=True, timeout=7200,
            )
            if r.returncode != 0:
                raise RuntimeError(f"clamp3_embd failed:\n{r.stdout[-2000:]}\n{r.stderr[-2000:]}")
            out = [np.load(os.path.join(outdir, s + ".npy")).reshape(-1) for s in stems]
            return _l2(np.stack(out))
        finally:
            shutil.rmtree(work, ignore_errors=True)

    def embed_audio(self, paths):
        return self._run(list(paths), ".wav", lambda src, dst: shutil.copy(src, dst))

    def embed_text(self, texts):
        def w(t, dst):
            with open(dst, "w") as f:
                f.write(t)
        return self._run(list(texts), ".txt", w)
