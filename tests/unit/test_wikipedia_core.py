"""The Wikipedia core that ships in the repository: present, well formed, loaded without re-embedding when the
model matches, and never trusted when it does not."""
import gzip
import json
import pathlib

import numpy as np
import pytest

from edge import seed

KNOW = pathlib.Path(__file__).resolve().parents[2] / "knowledge"
CORE = KNOW / "simple_wikipedia_core.jsonl.gz"
VEC = KNOW / "simple_wikipedia_core.vec.npz"


def _rows():
    with gzip.open(CORE, "rt", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh]


def test_core_ships_with_the_repository_and_cites_every_article():
    rows = _rows()
    # tools/repair_wikipedia_core.py drops the few leads that cannot be repaired, so the core is "about 30,000"
    assert seed.WIKI_CORE - 100 <= len(rows) <= seed.WIKI_CORE
    assert all(r["url"].startswith("https://simple.wikipedia.org/wiki/") and r["title"] and len(r["text"]) >= 40
               for r in rows)
    assert len({r["title"] for r in rows}) == len(rows), "ref ids are built from titles, so they must be unique"
    assert (KNOW / "SIMPLE_WIKIPEDIA_LICENSE.md").exists(), "CC BY-SA needs its attribution notice beside the data"


def test_vectors_ship_with_the_core_and_line_up_with_it():
    blob = np.load(VEC, allow_pickle=False)
    assert blob["vecs"].shape == (len(_rows()), 384) and blob["vecs"].dtype == np.float16
    assert str(blob["model"]) == "BAAI/bge-small-en-v1.5"
    assert np.isfinite(blob["vecs"]).all() and (np.abs(blob["vecs"]).sum(axis=1) > 0).all()


def test_a_sample_of_shipped_vectors_matches_the_real_model():
    """Guards the alignment of rows with vectors: if they drifted, a search would return the wrong article."""
    from shared.embed import load_embedder
    emb = load_embedder()
    if emb.name != "BAAI/bge-small-en-v1.5":
        pytest.skip("the real text model is not installed")
    rows, blob = _rows(), np.load(VEC, allow_pickle=False)
    for i in (0, 4_999, 14_999, len(rows) - 1):
        v = np.asarray(emb.embed_documents([f"{rows[i]['title']}: {rows[i]['text']}"])[0], dtype=np.float32)
        s = blob["vecs"][i].astype(np.float32)
        assert float(v @ s / (np.linalg.norm(v) * np.linalg.norm(s))) > 0.999, f"row {i} does not match its vector"


def _tiny_core(tmp_path, model):
    know = tmp_path / "knowledge"
    know.mkdir()
    with gzip.open(know / "simple_wikipedia_core.jsonl.gz", "wt", encoding="utf-8") as g:
        for t in ("Alpha", "Beta"):
            g.write(json.dumps({"title": t, "text": f"{t} is a letter of the Greek alphabet used in many fields.",
                                "url": f"https://simple.wikipedia.org/wiki/{t}"}) + "\n")
    np.savez_compressed(know / "simple_wikipedia_core.vec.npz", vecs=np.ones((2, 384), dtype=np.float16),
                        model=np.array(model))
    return know


def test_shipped_vectors_are_used_without_embedding_when_the_model_matches(make_device, monkeypatch, tmp_path):
    d, _ = make_device("solo", "site1")
    monkeypatch.setattr(seed, "KNOWLEDGE", _tiny_core(tmp_path, d.embedder.name))

    def forbidden(texts):
        raise AssertionError("the shipped vectors should have been used; nothing needed embedding")
    monkeypatch.setattr(d.embedder, "embed_documents", forbidden)
    assert seed.seed_if_empty(d, ["simple_wikipedia"])["seeded"] == 2


def test_a_different_model_never_reuses_the_shipped_vectors(make_device, monkeypatch, tmp_path):
    d, _ = make_device("solo", "site1")
    monkeypatch.setattr(seed, "KNOWLEDGE", _tiny_core(tmp_path, "some-other-model"))
    calls = []
    real = d.embedder.embed_documents
    monkeypatch.setattr(d.embedder, "embed_documents", lambda texts: calls.append(len(texts)) or real(texts))
    assert seed.seed_if_empty(d, ["simple_wikipedia"])["seeded"] == 2
    assert calls == [2], "vectors from another model are not comparable, so the texts must be embedded again"
