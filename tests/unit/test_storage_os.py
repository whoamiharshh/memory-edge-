"""OS compression of the device folder: Edge's pre-allocated zero pages shrink to almost nothing, and the shard keeps
working (writes, reopen, hybrid search)."""
import os

import numpy as np
import pytest

from edge.storage_os import allocated_bytes, enable_compression
from edge.store_edge import EdgeStore, StorePoint
from shared import ids


@pytest.mark.skipif(os.name != "nt", reason="NTFS compression is Windows-only")
def test_compressing_an_open_shard_shrinks_it_and_it_keeps_working(tmp_path):
    root = tmp_path / "dev"
    s = EdgeStore(root / "local")
    rng = np.random.default_rng(0)
    pts = lambda tag: [StorePoint(ids.make_id(tag, i), {"type": "episode"}, vib=rng.normal(size=27).tolist(),
                                  note=rng.normal(size=384).tolist(), bm25_text=f"{tag} noise case {i}") for i in range(300)]
    try:
        s.upsert(pts("bearing"))
        before = allocated_bytes(root)
        assert enable_compression(root)["compressed"] is True        # while the shard is open (as the device does)
        assert allocated_bytes(root) < before / 20
        s.upsert(pts("pump"))                                          # writes keep working
        assert s.search(text="pump noise case 42", limit=3)
    finally:
        s.close()
    s = EdgeStore(root / "local")
    try:
        assert s.count() == 600
    finally:
        s.close()


def test_non_windows_is_a_labelled_noop(tmp_path, monkeypatch):
    monkeypatch.setattr(os, "name", "posix")
    r = enable_compression(tmp_path / "x")
    assert r["compressed"] is False and "sparse" in r["reason"]
