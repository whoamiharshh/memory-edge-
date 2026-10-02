"""Extractive question answering with a "no answer" option, fully offline (ONNX Runtime, no torch).

Given a question and a passage, find the span of the passage that answers it - or decide the passage does not. The model
is deepset/roberta-base-squad2 (trained on SQuAD 2.0, which includes unanswerable questions), converted to ONNX and int8
quantized (models_cache/qa/roberta-base-squad2, fetched from onnx-community/roberta-base-squad2-ONNX).

Why this and not a generative model: it can only return text that is IN the passage, so it cannot invent a name or a
number, and it is fast (tens of milliseconds for a Wikipedia lead on a laptop CPU, against seconds for a 1.5B model).
`margin` is how much better the best span scores than "no answer"; the caller picks the threshold, because that is the
dial between answering more questions and answering only the ones it is sure of.
"""
from __future__ import annotations

import pathlib
import re
import threading

import numpy as np

DIR = pathlib.Path(__file__).resolve().parents[1] / "models_cache" / "qa" / "roberta-base-squad2"
MODEL = DIR / "onnx" / "model_quantized.onnx"
MAX_ANSWER_TOKENS = 30
MAX_LEN = 512

_lock = threading.Lock()
_state: dict = {}


def available() -> bool:
    return MODEL.exists() and (DIR / "tokenizer.json").exists()


def _load():
    with _lock:
        if not _state:
            import onnxruntime as ort
            from tokenizers import Tokenizer
            tok = Tokenizer.from_file(str(DIR / "tokenizer.json"))
            tok.enable_truncation(max_length=MAX_LEN, strategy="only_second")
            so = ort.SessionOptions()
            so.intra_op_num_threads = 4
            _state["tok"] = tok
            _state["sess"] = ort.InferenceSession(str(MODEL), sess_options=so, providers=["CPUExecutionProvider"])
    return _state["tok"], _state["sess"]


def _sentence(passage: str, a: int, b: int) -> str:
    """The sentence of `passage` that contains the character span [a, b), verbatim."""
    start = max([m.end() for m in re.finditer(r"(?<=[.!?])\s+", passage[:a])] or [0])
    m = re.search(r"(?<=[.!?])\s", passage[b:])
    end = b + m.start() if m else len(passage)
    return passage[start:end].strip()


def answer(question: str, passage: str) -> dict | None:
    """Best answer span in `passage`, as {"text", "sentence", "margin"}; None if the passage cannot be read.

    margin = (score of the best span) - (score of "no answer"). Above 0 the model prefers an answer; the higher, the
    more sure it is."""
    if not question.strip() or not passage.strip():
        return None
    tok, sess = _load()
    enc = tok.encode(question, passage)
    ids = np.asarray([enc.ids], dtype=np.int64)
    out = sess.run(None, {"input_ids": ids, "attention_mask": np.ones_like(ids)})
    start, end = out[0][0], out[1][0]
    ctx = np.asarray([s == 1 for s in enc.sequence_ids], dtype=bool)
    null = float(start[0] + end[0])
    best, span = -1e9, None
    idx = np.where(ctx)[0]
    for i in idx:
        # only spans that start in the passage and are short; the end scores are scanned in a small window
        js = idx[(idx >= i) & (idx < i + MAX_ANSWER_TOKENS)]
        if not len(js):
            continue
        j = js[int(np.argmax(end[js]))]
        sc = float(start[i] + end[j])
        if sc > best:
            best, span = sc, (int(i), int(j))
    if span is None:
        return None
    a, b = enc.offsets[span[0]][0], enc.offsets[span[1]][1]
    text = passage[a:b].strip()
    if not text:
        return None
    return {"text": text, "sentence": _sentence(passage, a, b), "margin": round(best - null, 3)}
