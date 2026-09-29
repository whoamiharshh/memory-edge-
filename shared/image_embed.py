"""Pictures as searchable memory, using Qdrant's own CLIP models through fastembed.

The point of this module is retrieval, not description. Nothing here ever says what is in a photograph.
It turns a picture into a vector and turns typed words into a vector in the *same* space, so Qdrant Edge
can answer "which pictures does this device already hold that look like this one?" — and the notes and
outcomes attached to those pictures are what actually answer the question.

That distinction is the whole reason to do it this way. A vision-language model would happily state that
a bearing is spalled, having never been trained on one, and the rest of this system refuses to claim more
than it measured. A nearest-neighbour hit makes no claim at all: a person looks at the retrieved pictures
and decides.

  Qdrant/clip-ViT-B-32-vision  340 MB  picture -> 512-d
  Qdrant/clip-ViT-B-32-text    250 MB  words   -> the same 512-d space

Both download once and then run offline on the CPU (~0.2 s per picture, measured on the build laptop).
Video is handled by sampling frames and embedding them as pictures; there is no video model.
"""
from __future__ import annotations

import os
import pathlib
import threading
from typing import Iterable, Sequence

VISION_MODEL = "Qdrant/clip-ViT-B-32-vision"
TEXT_MODEL = "Qdrant/clip-ViT-B-32-text"
DIM = 512
CACHE = pathlib.Path("models_cache") / "fastembed"


def _normalise(v: Sequence[float]) -> list[float]:
    """CLIP similarity is cosine and the shard stores cosine vectors, so unit-length them once here."""
    out = [float(x) for x in v]
    n = sum(x * x for x in out) ** 0.5
    return [x / n for x in out] if n else out


class ImageEmbedder:
    """Loads the two CLIP halves on first use.

    Loading costs about two minutes the very first time (the download) and a few seconds afterwards, so it
    is deliberately lazy: a device that never stores a picture never pays for it. The text and image halves
    load independently, because searching by words does not need the vision half.
    """

    def __init__(self, cache_dir: str | os.PathLike | None = None):
        self.cache_dir = str(cache_dir or CACHE)
        self._vision = None
        self._text = None
        self._lock = threading.Lock()
        self._failed: str | None = None

    @property
    def available(self) -> bool:
        """True when fastembed is importable. It says nothing about whether the weights are on disk yet:
        the first call downloads them, which is a slow call, not an unavailable one."""
        if self._failed:
            return False
        try:
            import fastembed  # noqa: F401
            return True
        except Exception as e:
            self._failed = f"{type(e).__name__}: {e}"
            return False

    def _load_vision(self):
        if self._vision is None:
            with self._lock:
                if self._vision is None:
                    from fastembed import ImageEmbedding
                    self._vision = ImageEmbedding(VISION_MODEL, cache_dir=self.cache_dir)
        return self._vision

    def _load_text(self):
        if self._text is None:
            with self._lock:
                if self._text is None:
                    from fastembed import TextEmbedding
                    self._text = TextEmbedding(TEXT_MODEL, cache_dir=self.cache_dir)
        return self._text

    def embed_images(self, paths: Iterable[str | os.PathLike]) -> list[list[float]]:
        paths = [str(p) for p in paths]
        if not paths:
            return []
        return [_normalise(v) for v in self._load_vision().embed(paths)]

    def embed_query(self, text: str) -> list[float]:
        """Words placed in the picture space, so typing "cracked housing" can retrieve photographs."""
        return _normalise(next(iter(self._load_text().embed([text]))))


def sample_video_frames(path: str | os.PathLike, every_seconds: float = 2.0,
                        max_frames: int = 12) -> list[str]:
    """Write out frames from a video so they can be embedded as ordinary pictures.

    There is no video model here and there does not need to be one: a fault visible at all is visible in a
    frame. Sampling is capped so a long clip cannot fill the disk or stall the device. Returns the frame
    paths, or an empty list when OpenCV is not installed.
    """
    try:
        import cv2
    except Exception:
        return []
    src = pathlib.Path(path)
    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        return []
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    step = max(1, int(fps * every_seconds))
    out_dir = src.parent / f"{src.stem}_frames"
    out_dir.mkdir(exist_ok=True)
    frames, i, kept = [], 0, 0
    try:
        while kept < max_frames:
            ok, frame = cap.read()
            if not ok:
                break
            if i % step == 0:
                fp = out_dir / f"{kept:03d}.jpg"
                cv2.imwrite(str(fp), frame)
                frames.append(str(fp))
                kept += 1
            i += 1
    finally:
        cap.release()
    return frames
