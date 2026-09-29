"""Speech to text, on the device.

The browser's own speech API would have been one line of JavaScript, but on most platforms it uploads the
audio to Google or Apple to be transcribed. This project claims to work with the network off and to keep
what a person says on their own device, and that claim has to survive the microphone button.

Vosk does the recognition locally: a 40 MB English model, Apache-2.0, CPU only, no torch. The audio never
leaves the machine, and the feature keeps working in a basement with no signal — which is exactly where
somebody is most likely to be talking to a machine instead of typing.

Accuracy is that of a small model: fine for short spoken notes, weaker on unusual proper nouns and in heavy
background noise. What it produces is always shown to the person as editable text before anything is
stored, so a misheard word gets corrected rather than silently remembered.
"""
from __future__ import annotations

import io
import json
import os
import pathlib
import threading
import wave

MODEL_DIR = pathlib.Path("models_cache") / "vosk" / "vosk-model-small-en-us-0.15"
MODEL_URL = "https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip"
TARGET_RATE = 16000
MAX_SECONDS = 120


class Recogniser:
    """Holds the Vosk model, loaded once on first use (about a second) and reused after that."""

    def __init__(self, model_dir: str | os.PathLike | None = None):
        self.model_dir = pathlib.Path(model_dir or MODEL_DIR)
        self._model = None
        self._lock = threading.Lock()

    @property
    def available(self) -> bool:
        """Both the library and the weights have to be present. Missing weights is the normal case on a
        fresh clone, and the UI says so rather than failing when the button is pressed."""
        try:
            import vosk  # noqa: F401
        except Exception:
            return False
        return (self.model_dir / "am").exists() or (self.model_dir / "conf").exists()

    def status(self) -> dict:
        try:
            import vosk  # noqa: F401
            lib = True
        except Exception:
            lib = False
        return {"available": self.available, "library": lib,
                "model_present": self.model_dir.exists(),
                "model_dir": str(self.model_dir), "download": MODEL_URL}

    def _load(self):
        if self._model is None:
            with self._lock:
                if self._model is None:
                    import vosk
                    vosk.SetLogLevel(-1)               # the C library is chatty on stderr otherwise
                    self._model = vosk.Model(str(self.model_dir))
        return self._model

    def transcribe_wav(self, data: bytes) -> dict:
        """Transcribe 16-bit PCM WAV bytes.

        Mono 16 kHz is what the model expects. Anything else is converted here rather than refused,
        because a browser records at whatever its hardware prefers and the person pressing the button
        should not have to care.
        """
        import vosk

        if not self.available:
            raise RuntimeError("offline speech model not installed; see status() for the download link")
        with wave.open(io.BytesIO(data), "rb") as w:
            if w.getsampwidth() != 2:
                raise ValueError("audio must be 16-bit PCM")
            frames, rate, channels = w.getnframes(), w.getframerate(), w.getnchannels()
            if frames / float(rate or 1) > MAX_SECONDS:
                raise ValueError(f"recording longer than {MAX_SECONDS}s")
            pcm = w.readframes(frames)

        pcm, rate = _to_mono_16k(pcm, rate, channels)
        rec = vosk.KaldiRecognizer(self._load(), rate)
        rec.SetWords(False)
        rec.AcceptWaveform(pcm)
        text = (json.loads(rec.FinalResult()).get("text") or "").strip()
        return {"text": text, "seconds": round(len(pcm) / 2 / rate, 1), "model": self.model_dir.name}


def _to_mono_16k(pcm: bytes, rate: int, channels: int) -> tuple[bytes, int]:
    """Down-mix and resample to what the model expects, using numpy rather than `audioop`, which was
    removed in Python 3.13."""
    import numpy as np

    x = np.frombuffer(pcm, dtype="<i2")
    if channels > 1:
        x = x.reshape(-1, channels).mean(axis=1)
    if rate != TARGET_RATE and rate > 0 and x.size:
        n = int(round(x.size * TARGET_RATE / rate))
        # linear resampling: the model's own front end band-limits the signal, so the artefacts a proper
        # anti-aliasing filter would remove do not survive as far as the acoustic features anyway
        x = np.interp(np.linspace(0, x.size - 1, n), np.arange(x.size), x.astype(np.float64))
    return np.asarray(x, dtype="<i2").tobytes(), TARGET_RATE
