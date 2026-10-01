"""Fetch or build the on-device language models. Run once per machine; everything is offline afterwards.

    .venv\\Scripts\\python.exe -m tools.build_language_models              # both
    .venv\\Scripts\\python.exe -m tools.build_language_models --whisper-only
    .venv\\Scripts\\python.exe -m tools.build_language_models --translation-only [--venv PATH] [--out DIR]

Speech (shared/speech.py): downloads faster-whisper `small` (~480 MB, MIT) into models_cache/whisper/small.

Translation (shared/translate.py): the Helsinki-NLP opus-mt hi-en and en-hi models exist only as PyTorch/ONNX
weights, so they are converted ONCE to CTranslate2 int8 (~78 MB each). The conversion needs torch and
transformers, which this project deliberately does not carry at run time, so it happens in a throwaway venv
that is deleted afterwards (or pass --venv to reuse one). The converted models and their SentencePiece
tokenizers land in models_cache/translate/<pair>/ and need only ctranslate2 + sentencepiece.
"""
from __future__ import annotations

import argparse
import pathlib
import shutil
import subprocess
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parent.parent
PAIRS = {"hi-en": "Helsinki-NLP/opus-mt-hi-en", "en-hi": "Helsinki-NLP/opus-mt-en-hi"}
CONVERT = ("import sys; from ctranslate2.converters.transformers import main; "
           "sys.argv = ['ct2-transformers-converter'] + sys.argv[1:]; main()")


def fetch_whisper() -> None:
    dest = ROOT / "models_cache" / "whisper" / "small"
    if (dest / "model.bin").exists():
        print(f"whisper: already at {dest}")
        return
    from faster_whisper import download_model
    print("whisper: downloading small (~480 MB)...")
    download_model("small", output_dir=str(dest))
    print(f"whisper: ready at {dest}")


def _venv_python(venv: pathlib.Path) -> pathlib.Path:
    return venv / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")


def _make_venv() -> pathlib.Path:
    venv = pathlib.Path(tempfile.mkdtemp(prefix="ct2conv_")) / "venv"
    subprocess.run(["uv", "venv", str(venv), "--python", "3.12"], check=True)
    py = str(_venv_python(venv))
    subprocess.run(["uv", "pip", "install", "--python", py, "torch", "--index-url",
                    "https://download.pytorch.org/whl/cpu"], check=True)
    subprocess.run(["uv", "pip", "install", "--python", py, "transformers", "sentencepiece", "ctranslate2",
                    "sacremoses"], check=True)
    return venv


def build_translation(venv: pathlib.Path | None, out: pathlib.Path) -> None:
    todo = [p for p in PAIRS if not (out / p / "model.bin").exists()]
    for p in PAIRS:
        if p not in todo:
            print(f"translation {p}: already at {out / p}")
    if todo:
        own = venv is None
        venv = venv or _make_venv()
        try:
            for pair in todo:
                print(f"translation {pair}: converting {PAIRS[pair]} (int8)...")
                subprocess.run([str(_venv_python(venv)), "-c", CONVERT, "--model", PAIRS[pair],
                                "--output_dir", str(out / pair), "--quantization", "int8"], check=True)
        finally:
            if own:
                shutil.rmtree(venv.parent, ignore_errors=True)
    # the converter does not copy the SentencePiece tokenizers, which the translator needs
    from huggingface_hub import hf_hub_download
    for pair, repo in PAIRS.items():
        for name in ("source.spm", "target.spm"):
            if not (out / pair / name).exists():
                shutil.copy(hf_hub_download(repo, name), out / pair / name)
    print(f"translation: ready at {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--whisper-only", action="store_true")
    ap.add_argument("--translation-only", action="store_true")
    ap.add_argument("--venv", type=pathlib.Path, help="reuse a venv that already has torch + transformers + ctranslate2")
    ap.add_argument("--out", type=pathlib.Path, default=ROOT / "models_cache" / "translate")
    a = ap.parse_args()
    if not a.translation_only:
        fetch_whisper()
    if not a.whisper_only:
        build_translation(a.venv, a.out)


if __name__ == "__main__":
    main()
