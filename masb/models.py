"""Contrastive music-text model registry. Each loader returns an object with

    .name
    .embed_audio(paths: list[str]) -> np.ndarray [N, D]   (L2-normalised)
    .embed_text(texts:  list[str]) -> np.ndarray [N, D]   (L2-normalised)

so the CLAP score s(a, c) is the row-wise dot product. Joint-space dimensions:
LAION-CLAP 512, MS-CLAP 2023 1024, MuQ-MuLan 512, CLaMP 3 768.

Imports are lazy so one missing dependency does not block the other models.
Qwen2-Audio (generative) is scored separately in exp1_qwen.py.
"""
import numpy as np

from masb.paths import LAION_CLAP_CKPT


def _l2(x):
    x = np.asarray(x, dtype=np.float32)
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-8)


class LaionClap:
    """LAION-CLAP, music checkpoint (HTSAT-base audio, RoBERTa text)."""
    name = "laion-clap-music"

    def __init__(self):
        import laion_clap
        if not LAION_CLAP_CKPT.exists():
            raise FileNotFoundError(f"LAION-CLAP checkpoint not found at {LAION_CLAP_CKPT}; set LAION_CLAP_CKPT")
        self.m = laion_clap.CLAP_Module(enable_fusion=False, amodel="HTSAT-base")
        self.m.load_ckpt(str(LAION_CLAP_CKPT))

    def embed_audio(self, paths, batch=16):
        paths = list(paths)
        out = [self.m.get_audio_embedding_from_filelist(x=paths[i:i + batch], use_tensor=False)
               for i in range(0, len(paths), batch)]
        return _l2(np.concatenate(out, axis=0))

    def embed_text(self, texts, batch=64):
        texts = list(texts)
        out = [self.m.get_text_embedding(texts[i:i + batch], use_tensor=False)
               for i in range(0, len(texts), batch)]
        return _l2(np.concatenate(out, axis=0))


class MsClap:
    """Microsoft CLAP 2023 (GPT-2 text encoder). Reads a centred 7 s of each clip."""
    name = "ms-clap-2023"

    def __init__(self):
        from msclap import CLAP
        self.m = CLAP(version="2023", use_cuda=True)

    def embed_audio(self, paths, batch=16):
        import torch
        paths = list(paths)
        outs = []
        for i in range(0, len(paths), batch):
            with torch.no_grad():
                e = self.m.get_audio_embeddings(paths[i:i + batch])
            outs.append(e.detach().cpu().numpy())
        return _l2(np.concatenate(outs, axis=0))

    def embed_text(self, texts, batch=64):
        import torch
        texts = list(texts)
        outs = []
        for i in range(0, len(texts), batch):
            with torch.no_grad():
                e = self.m.get_text_embeddings(texts[i:i + batch])
            outs.append(e.detach().cpu().numpy())
        return _l2(np.concatenate(outs, axis=0))


class MuqMulan:
    """MuQ-MuLan (OpenMuQ/MuQ-MuLan-large; XLM-RoBERTa text encoder)."""
    name = "muq-mulan"

    def __init__(self):
        import torch
        from muq import MuQMuLan
        self.dev = "cuda" if torch.cuda.is_available() else "cpu"
        self.m = MuQMuLan.from_pretrained("OpenMuQ/MuQ-MuLan-large").to(self.dev).eval()

    def embed_audio(self, paths):
        import torch, librosa
        outs = []
        for p in paths:
            wav, _ = librosa.load(p, sr=24000, mono=True)
            with torch.no_grad():
                e = self.m(wavs=torch.tensor(wav).unsqueeze(0).to(self.dev))
            outs.append(e.detach().cpu().numpy()[0])
        return _l2(np.stack(outs))

    def embed_text(self, texts):
        import torch
        with torch.no_grad():
            e = self.m(texts=list(texts))
        return _l2(e.detach().cpu().numpy())


def _clamp3():
    from masb.clamp3_model import Clamp3
    return Clamp3()


REGISTRY = {
    "laion-clap": LaionClap,
    "ms-clap": MsClap,
    "muq-mulan": MuqMulan,
    "clamp3": _clamp3,  # shells out to its own environment, see clamp3_model.py
}
