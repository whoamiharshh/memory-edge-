"""Speech to text, on the device.

The browser's own speech API would have been one line of JavaScript, but on most platforms it uploads the
audio to Google or Apple to be transcribed. This project claims to work with the network off and to keep
what a person says on their own device, and that claim has to survive the microphone button.

Whisper (faster-whisper / CTranslate2, MIT, CPU only, no torch) does the recognition locally. It is
multilingual and detects the language itself, so a person can speak English or Hindi and the words are
written in the language they spoke, in its own script. The `small` model is about 480 MB, downloaded once.
The audio never leaves the machine and the feature keeps working with the network off.

The 40 MB English-only Vosk model this used to run on is kept as a fallback for a device that has not
downloaded the Whisper weights. It cannot recognise Hindi, and it struggles with accented English.

What is produced is always shown to the person as editable text before anything is stored, so a misheard
word gets corrected rather than silently remembered.
"""
from __future__ import annotations

import io
import json
import os
import pathlib
import threading
import wave

WHISPER_DIR = pathlib.Path("models_cache") / "whisper" / "small"
WHISPER_REPO = "Systran/faster-whisper-small"
MODEL_DIR = pathlib.Path("models_cache") / "vosk" / "vosk-model-small-en-us-0.15"
MODEL_URL = "https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip"
TARGET_RATE = 16000
MAX_SECONDS = 120
# Whisper invents text on silence and on near-silence. Anything below this level is treated as nothing said.
MIN_RMS = 0.003


class Recogniser:
    """Holds the speech model, loaded once on first use and reused after that."""

    def __init__(self, model_dir: str | os.PathLike | None = None,
                 whisper_dir: str | os.PathLike | None = None):
        self.model_dir = pathlib.Path(model_dir or MODEL_DIR)
        self.whisper_dir = pathlib.Path(whisper_dir or WHISPER_DIR)
        self._model = None
        self._whisper = None
        self._lock = threading.Lock()

    # ---- what is installed --------------------------------------------------------------------------
    @property
    def whisper_available(self) -> bool:
        try:
            import faster_whisper  # noqa: F401
        except Exception:
            return False
        return (self.whisper_dir / "model.bin").exists()

    @property
    def vosk_available(self) -> bool:
        """Both the library and the weights have to be present. Missing weights is the normal case on a
        fresh clone, and the UI says so rather than failing when the button is pressed."""
        try:
            import vosk  # noqa: F401
        except Exception:
            return False
        return (self.model_dir / "am").exists() or (self.model_dir / "conf").exists()

    @property
    def available(self) -> bool:
        return self.whisper_available or self.vosk_available

    def status(self) -> dict:
        try:
            import vosk  # noqa: F401
            lib = True
        except Exception:
            lib = False
        return {"available": self.available, "library": lib,
                "engine": "whisper" if self.whisper_available else ("vosk" if self.vosk_available else None),
                "multilingual": self.whisper_available,
                "model_present": self.model_dir.exists() or self.whisper_dir.exists(),
                "model_dir": str(self.whisper_dir if self.whisper_available else self.model_dir),
                "download": MODEL_URL if self.whisper_available else f"https://huggingface.co/{WHISPER_REPO}"}

    # ---- loading --------------------------------------------------------------------------------------
    def _load_vosk(self):
        if self._model is None:
            with self._lock:
                if self._model is None:
                    import vosk
                    vosk.SetLogLevel(-1)               # the C library is chatty on stderr otherwise
                    self._model = vosk.Model(str(self.model_dir))
        return self._model

    def _load_whisper(self):
        if self._whisper is None:
            with self._lock:
                if self._whisper is None:
                    from faster_whisper import WhisperModel
                    self._whisper = WhisperModel(str(self.whisper_dir), device="cpu", compute_type="int8",
                                                 cpu_threads=max(1, (os.cpu_count() or 4) - 2))
        return self._whisper

    # ---- transcription --------------------------------------------------------------------------------
    def transcribe_wav(self, data: bytes, language: str | None = None) -> dict:
        """Transcribe 16-bit PCM WAV bytes.

        Mono 16 kHz is what the models expect. Anything else is converted here rather than refused,
        because a browser records at whatever its hardware prefers and the person pressing the button
        should not have to care. `language` is an ISO code ("en", "hi") to force one; left empty, the
        language is detected from the audio.
        """
        if not self.available:
            raise RuntimeError("no offline speech model installed; see status() for the download link")
        with wave.open(io.BytesIO(data), "rb") as w:
            if w.getsampwidth() != 2:
                raise ValueError("audio must be 16-bit PCM")
            frames, rate, channels = w.getnframes(), w.getframerate(), w.getnchannels()
            if frames / float(rate or 1) > MAX_SECONDS:
                raise ValueError(f"recording longer than {MAX_SECONDS}s")
            pcm = w.readframes(frames)

        pcm, rate = _to_mono_16k(pcm, rate, channels)
        seconds = round(len(pcm) / 2 / rate, 1)
        if self.whisper_available:
            return self._whisper_text(pcm, seconds, language)
        return self._vosk_text(pcm, rate, seconds)

    def _whisper_text(self, pcm: bytes, seconds: float, language: str | None) -> dict:
        import numpy as np

        audio = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0
        if audio.size == 0 or float(np.sqrt(np.mean(audio ** 2))) < MIN_RMS:
            return {"text": "", "seconds": seconds, "model": "whisper-small", "language": None}
        # a numpy array goes straight in: faster-whisper's own file decoder is not needed, and not used
        segments, info = self._load_whisper().transcribe(
            audio, task="transcribe", language=language or None, beam_size=3,
            vad_filter=True, condition_on_previous_text=False)
        text = " ".join(s.text.strip() for s in segments).strip()
        return {"text": text, "seconds": seconds, "model": "whisper-small",
                "language": info.language, "language_probability": round(float(info.language_probability), 2)}

    def _vosk_text(self, pcm: bytes, rate: int, seconds: float) -> dict:
        import vosk

        rec = vosk.KaldiRecognizer(self._load_vosk(), rate)
        rec.SetWords(False)
        rec.AcceptWaveform(pcm)
        text = (json.loads(rec.FinalResult()).get("text") or "").strip()
        return {"text": text, "seconds": seconds, "model": self.model_dir.name, "language": "en"}


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
