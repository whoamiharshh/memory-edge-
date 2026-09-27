"""Text embedders. BgeEmbedder is the real model (BAAI/bge-small-en-v1.5, 384-d, MIT, via FastEmbed, runs
fully offline once the model files are in models_cache/). HashEmbedder is a deterministic stand-in used by
fast unit tests and as an explicit, labelled fallback; it is NOT a semantic model."""
from __future__ import annotations

import hashlib
import os
import pathlib
import re
import threading
from typing import Protocol, Sequence

import numpy as np

MODEL_NAME = "BAAI/bge-small-en-v1.5"
DIM = 384
CACHE = pathlib.Path(__file__).resolve().parents[1] / "models_cache"


class Embedder(Protocol):
    name: str
    dim: int

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]: ...
    def embed_query(self, text: str) -> list[float]: ...


class FastEmbedder:
    """Any FastEmbed dense text model (CPU, ONNX). offline=True never touches the network: the model files must
    already be in cache_dir (download once with offline=False)."""

    def __init__(self, model_name: str = MODEL_NAME, dim: int = DIM, cache_dir: str | os.PathLike = CACHE,
                 offline: bool = True):
        from fastembed import TextEmbedding
        os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
        if offline:
            os.environ["HF_HUB_OFFLINE"] = "1"      # never reach the network at runtime (edge is offline-first)
        self.name, self.dim = model_name, dim
        self._model = TextEmbedding(model_name, cache_dir=str(cache_dir))
        self._lock = threading.Lock()

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        with self._lock:
            return [v.tolist() for v in self._model.passage_embed(list(texts))]

    def embed_query(self, text: str) -> list[float]:
        with self._lock:
            return next(iter(self._model.query_embed(text))).tolist()


class BgeEmbedder(FastEmbedder):
    """The shipped model: BAAI/bge-small-en-v1.5 (384-d, MIT). Chosen over MiniLM by bench/embed_models.py."""

    def __init__(self, cache_dir: str | os.PathLike = CACHE, offline: bool = True):
        super().__init__(MODEL_NAME, DIM, cache_dir, offline)


class HashEmbedder:
    """Hashed bag-of-words, L2-normalised. Deterministic, dependency-free, lexical only."""
    name = "hash-bow-384 (test stand-in, not semantic)"
    dim = DIM

    def _vec(self, text: str) -> list[float]:
        v = np.zeros(DIM)
        for tok in re.findall(r"[a-z0-9]+", text.lower()):
            h = int.from_bytes(hashlib.md5(tok.encode()).digest()[:4], "little")
            v[h % DIM] += 1.0
        n = np.linalg.norm(v)
        if n == 0:
            v[0] = 1.0
            n = 1.0
        return (v / n).tolist()

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._vec(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vec(text)


def load_embedder(prefer_real: bool = True) -> Embedder:
    if prefer_real:
        try:
            return BgeEmbedder()
        except Exception as e:   # model files missing: say so loudly, fall back explicitly
            print(f"[embed] bge-small unavailable ({e!r}); using HashEmbedder fallback")
    return HashEmbedder()
