"""Manufacturer manuals indexed and searched offline, with page citations. The PDF here is built by the test (a
tiny text-only fixture), so no copyrighted manual is needed."""
import base64

import pytest
from fastapi.testclient import TestClient

from edge import manuals
from edge.api import create_app
from edge.sync_worker import SyncWorker

H = {"X-Operator-Token": "op-1234567"}


def tiny_pdf(pages: list[str]) -> bytes:
    """A minimal valid PDF, one text line per page (Helvetica)."""
    objs = ["<< /Type /Catalog /Pages 2 0 R >>", None, "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"]
    kids = []
    for text in pages:
        esc = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
        stream = f"BT /F1 10 Tf 40 700 Td ({esc}) Tj ET"
        objs.append(f"<< /Length {len(stream)} >>\nstream\n{stream}\nendstream")
        content_no = len(objs)
        objs.append(f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 3 0 R >> >> "
                    f"/Contents {content_no} 0 R >>")
        kids.append(f"{len(objs)} 0 R")
    objs[1] = f"<< /Type /Pages /Kids [{' '.join(kids)}] /Count {len(kids)} >>"
    out, offs = b"%PDF-1.4\n", []
    for i, o in enumerate(objs, 1):
        offs.append(len(out))
        out += f"{i} 0 obj\n{o}\nendobj\n".encode("latin-1")
    xref = len(out)
    out += f"xref\n0 {len(objs) + 1}\n0000000000 65535 f \n".encode()
    out += "".join(f"{o:010d} 00000 n \n" for o in offs).encode()
    out += f"trailer\n<< /Size {len(objs) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF".encode()
    return out


PAGES = ["Safety: lock out the motor before any maintenance work on the drive end.",
         "Lubrication: relubricate the drive end bearing with 5 g of lithium grease every 2000 operating hours.",
         "Bearing replacement: use an induction heater to mount the new 6203 bearing, never hammer the outer ring."]


def test_manual_is_indexed_and_search_cites_the_page(make_device):
    d, _ = make_device("devA", "site1")
    doc = manuals.add(d, "Motor XYZ manual rev 3", tiny_pdf(PAGES), "maker's website")
    assert doc["pages"] == 3 and doc["passages"] == 3
    hits = manuals.search(d, "how much grease and how often")
    assert hits[0]["page"] == 2 and "2000 operating hours" in hits[0]["text"]
    hits = manuals.search(d, "bearing replacement heater")
    assert hits[0]["page"] == 3 and hits[0]["title"] == "Motor XYZ manual rev 3"
    assert manuals.search(d, "grease")[0]["doc_id"] == doc["doc_id"]
    with pytest.raises(ValueError, match="already indexed"):
        manuals.add(d, "again", tiny_pdf(PAGES))
    assert manuals.remove(d, doc["doc_id"]) and manuals.search(d, "grease") == []
    assert d.store.count({"type": "manual"}) == 0


@pytest.mark.parametrize("blob,msg", [(b"hello", "not a PDF"), (b"%PDF-1.4 garbage", "unreadable|no text"),
                                      (tiny_pdf([""]), "no text")])
def test_bad_manuals_are_refused(make_device, blob, msg):
    d, _ = make_device("devA", "site1")
    with pytest.raises(ValueError, match=msg):
        manuals.add(d, "bad manual", blob)


def test_manual_routes_and_episode_lookup(make_device):
    d, _ = make_device("devA", "site1")
    c = TestClient(create_app(d, SyncWorker(d, None, None), "op-1234567"))
    body = {"title": "Motor XYZ manual", "pdf_b64": base64.b64encode(tiny_pdf(PAGES)).decode()}
    assert c.post("/api/manuals", json=body).status_code == 401
    r = c.post("/api/manuals", json=body, headers=H)
    assert r.status_code == 200, r.text
    assert c.get("/api/manuals", headers=H).json()[0]["pages"] == 3
    s = c.get("/api/manuals/search", params={"q": "grease interval"}, headers=H).json()
    assert s["hits"][0]["page"] == 2
    assert manuals.query_for({"component": "bearing", "fault_class": "overheating"}).startswith("bearing lubrication")
    assert c.post("/api/manuals", json=body | {"pdf_b64": "!!"}, headers=H).status_code == 422
    assert c.delete(f"/api/manuals/{r.json()['doc_id']}", headers=H).status_code == 200
