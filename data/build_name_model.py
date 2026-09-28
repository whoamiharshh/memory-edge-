"""Train knowledge/name_model.json (shared/name_model.py) on the full real name list and domain vocabulary.
Run after data/fetch_names.py and data/build_vocab.py:  .venv\\Scripts\\python.exe -m data.build_name_model
"""
import json
import pathlib

from shared import name_model

ROOT = pathlib.Path(__file__).resolve().parents[1]


def main() -> None:
    names = list(json.loads((ROOT / "knowledge" / "given_names.json").read_text(encoding="utf-8"))["names"])
    words = json.loads((ROOT / "knowledge" / "domain_vocab.json").read_text(encoding="utf-8"))["words"]
    m = name_model.train(names, words)
    m["about"] = ("Character n-gram 'looks like a given name' model for the note redactor; trained on Wikidata given "
                  "names (CC0) vs maintenance vocabulary (see knowledge/domain_vocab.json). Threshold chosen on a "
                  "validation split.")
    (ROOT / "knowledge" / "name_model.json").write_text(json.dumps(m, separators=(",", ":")))
    print("threshold", m["threshold"], "validation", m["validation"])


if __name__ == "__main__":
    main()
