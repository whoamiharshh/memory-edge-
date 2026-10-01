#!/usr/bin/env sh
set -eu
ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
cd "$ROOT"
PYTHON_BIN="${PYTHON_BIN:-python3}"
command -v "$PYTHON_BIN" >/dev/null 2>&1 || { echo "Python 3.12+ is required." >&2; exit 1; }
"$PYTHON_BIN" - <<'PY'
import sys
if sys.version_info < (3, 12):
    raise SystemExit(f"Python 3.12+ is required; detected {sys.version.split()[0]}")
PY
if [ ! -x .venv/bin/python ]; then
  echo "Creating .venv..."
  "$PYTHON_BIN" -m venv .venv
fi
VENV_PY="$ROOT/.venv/bin/python"
"$VENV_PY" -m pip install --upgrade pip
# llama-cpp-python (the optional evidence-brief model) ships prebuilt CPU wheels on the author's own
# index. Without it pip falls back to building from source, which needs a C toolchain nobody was told
# to install, so setup failed on a clean machine.
"$VENV_PY" -m pip install -r requirements.txt \
  --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu
mkdir -p runtime models_cache backups
"$VENV_PY" -m tools.qdrant_local download
"$VENV_PY" -c "from fastembed import TextEmbedding; TextEmbedding('BAAI/bge-small-en-v1.5', cache_dir='models_cache')"
echo "Setup complete. Start with: ./.venv/bin/python -m demo.scenario after launching the services."
